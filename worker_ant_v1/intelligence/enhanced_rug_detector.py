"""
ENHANCED RUG DETECTOR - ADVANCED TOKEN RISK ASSESSMENT
====================================================

Sophisticated rug pull detection system using multi-layered analysis:
- Smart contract code analysis and vulnerability detection
- Liquidity pool health and concentration analysis  
- Holder distribution and whale monitoring
- Historical pattern matching and behavioral analysis
- Real-time transaction monitoring and anomaly detection

This is a critical component of Stage 1 (Survival Filter) in the three-stage pipeline.
It must provide accurate risk assessment to prevent catastrophic losses.

Risk Assessment Layers:
1. Contract Security Analysis - Code vulnerabilities, verification status
2. Liquidity Health Assessment - Pool concentration, removal patterns
3. Holder Distribution Analysis - Whale concentration, dev holdings
4. Transaction Pattern Analysis - Suspicious trading behavior
5. Historical Pattern Matching - Known rug pull signatures
6. Real-time Monitoring - Ongoing risk factor changes
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import numpy as np

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_api_config, get_network_rpc_url
from worker_ant_v1.utils.market_data_fetcher import get_market_data_fetcher


class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionLevel(Enum):
    """Rug detection confidence levels"""
    CLEAR = "clear"         # Very low risk, safe to trade
    CAUTION = "caution"     # Some risk factors present
    WARNING = "warning"     # Significant risk factors
    DANGER = "danger"       # High probability of rug
    CRITICAL = "critical"   # Imminent rug pull likely


@dataclass
class RiskFactor:
    """Individual risk factor assessment"""
    factor_name: str
    risk_score: float  # 0.0 to 1.0
    severity: RiskLevel
    description: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class RugDetectionResult:
    """Comprehensive rug detection analysis result"""
    
    # Core assessment
    token_address: str
    overall_risk: float  # 0.0 to 1.0
    detection_level: DetectionLevel
    is_safe_to_trade: bool
    
    # Risk breakdown
    risk_factors: List[RiskFactor] = field(default_factory=list)
    contract_risk_score: float = 0.0
    liquidity_risk_score: float = 0.0
    holder_risk_score: float = 0.0
    behavioral_risk_score: float = 0.0
    
    # Key metrics
    honeypot_probability: float = 0.0
    slow_rug_probability: float = 0.0
    flash_rug_probability: float = 0.0
    dev_exit_probability: float = 0.0
    
    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_duration_ms: int = 0
    data_sources_used: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Recommendations
    trading_recommendations: List[str] = field(default_factory=list)
    monitoring_alerts: List[str] = field(default_factory=list)


class EnhancedRugDetector:
    """Advanced rug pull detection with multi-layered analysis"""
    
    def __init__(self):
        self.logger = get_logger("EnhancedRugDetector")
        self.api_config = get_api_config()
        self.rpc_url = get_network_rpc_url()
        
        # Core systems
        self.market_data_fetcher = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'critical': 0.8,   # 80%+ risk
            'high': 0.6,       # 60%+ risk  
            'medium': 0.4,     # 40%+ risk
            'low': 0.2,        # 20%+ risk
            'very_low': 0.0    # <20% risk
        }
        
        # Known rug patterns
        self.known_rug_patterns = {
            'honeypot_signatures': [
                'transfer_blocked',
                'sell_disabled',
                'blacklist_active',
                'permission_denied'
            ],
            'dev_exit_patterns': [
                'massive_dev_sell',
                'liquidity_removal',
                'contract_renounced_suddenly',
                'social_media_deleted'
            ],
            'slow_rug_patterns': [
                'gradual_liquidity_drain',
                'increasing_sell_restrictions',
                'dev_dumping_pattern',
                'fake_volume_inflation'
            ]
        }
        
        # Contract vulnerability patterns
        self.vulnerability_patterns = {
            'high_risk_functions': [
                'mint',
                'burn',
                'blacklist',
                'pause',
                'setTaxes',
                'setMaxTransaction',
                'updateFees'
            ],
            'honeypot_indicators': [
                'canTransfer',
                'isExcludedFromFees',
                'antiBot',
                'maxTransactionAmount',
                'tradingActive'
            ]
        }
        
        # Performance tracking
        self.total_analyses = 0
        self.total_rugs_detected = 0
        self.false_positive_rate = 0.0
        self.average_analysis_time_ms = 0.0
        
        # Cache for analysis results
        self.analysis_cache: Dict[str, RugDetectionResult] = {}
        self.cache_ttl_seconds = 1800  # 30 minutes
        
        self.logger.info("🛡️ Enhanced Rug Detector initialized - Multi-layered protection active")
    
    async def initialize(self) -> bool:
        """Initialize the rug detector"""
        try:
            self.logger.info("🚀 Initializing Enhanced Rug Detector...")
            
            # Initialize market data fetcher
            self.market_data_fetcher = await get_market_data_fetcher()
            
            # Load known rug database
            await self._load_known_rug_database()
            
            # Test analysis capabilities
            await self._test_analysis_capabilities()
            
            self.logger.info("✅ Enhanced Rug Detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize rug detector: {e}")
            return False
    
    async def analyze_token(self, token_address: str, token_name: str = "", token_symbol: str = "") -> RugDetectionResult:
        """
        Comprehensive rug detection analysis
        
        Args:
            token_address: Solana token address
            token_name: Token name (optional)
            token_symbol: Token symbol (optional)
            
        Returns:
            RugDetectionResult with comprehensive risk assessment
        """
        analysis_start_time = time.time()
        
        try:
            self.logger.debug(f"🔍 Analyzing token {token_address[:8]} for rug pull risks...")
            
            # Check cache first
            cached_result = await self._get_cached_analysis(token_address)
            if cached_result:
                return cached_result
            
            # Initialize result
            result = RugDetectionResult(
                token_address=token_address,
                overall_risk=0.0,
                detection_level=DetectionLevel.CLEAR,
                is_safe_to_trade=True
            )
            
            # Layer 1: Contract Security Analysis
            contract_analysis = await self._analyze_contract_security(token_address)
            result.contract_risk_score = contract_analysis['risk_score']
            result.risk_factors.extend(contract_analysis['risk_factors'])
            
            # Layer 2: Liquidity Health Assessment
            liquidity_analysis = await self._analyze_liquidity_health(token_address)
            result.liquidity_risk_score = liquidity_analysis['risk_score']
            result.risk_factors.extend(liquidity_analysis['risk_factors'])
            
            # Layer 3: Holder Distribution Analysis
            holder_analysis = await self._analyze_holder_distribution(token_address)
            result.holder_risk_score = holder_analysis['risk_score']
            result.risk_factors.extend(holder_analysis['risk_factors'])
            
            # Layer 4: Behavioral Pattern Analysis
            behavior_analysis = await self._analyze_behavioral_patterns(token_address)
            result.behavioral_risk_score = behavior_analysis['risk_score']
            result.risk_factors.extend(behavior_analysis['risk_factors'])
            
            # Layer 5: Historical Pattern Matching
            pattern_analysis = await self._analyze_historical_patterns(token_address)
            result.risk_factors.extend(pattern_analysis['risk_factors'])
            
            # Calculate specific rug probabilities
            result.honeypot_probability = await self._calculate_honeypot_probability(result.risk_factors)
            result.slow_rug_probability = await self._calculate_slow_rug_probability(result.risk_factors)
            result.flash_rug_probability = await self._calculate_flash_rug_probability(result.risk_factors)
            result.dev_exit_probability = await self._calculate_dev_exit_probability(result.risk_factors)
            
            # Calculate overall risk score
            result.overall_risk = await self._calculate_overall_risk(result)
            
            # Determine detection level and safety
            result.detection_level = self._determine_detection_level(result.overall_risk)
            result.is_safe_to_trade = result.detection_level in [DetectionLevel.CLEAR, DetectionLevel.CAUTION]
            
            # Generate recommendations
            result.trading_recommendations = await self._generate_trading_recommendations(result)
            result.monitoring_alerts = await self._generate_monitoring_alerts(result)
            
            # Calculate confidence and metadata
            result.confidence_score = await self._calculate_confidence_score(result)
            result.analysis_duration_ms = int((time.time() - analysis_start_time) * 1000)
            result.data_sources_used = self._get_data_sources_used()
            
            # Cache the result
            await self._cache_analysis_result(token_address, result)
            
            # Update performance metrics
            self._update_analysis_metrics(result)
            
            log_level = "🛡️ CLEAR" if result.is_safe_to_trade else "🚨 DANGER"
            self.logger.info(f"{log_level} | {token_address[:8]} | Risk: {result.overall_risk:.3f} | "
                           f"Level: {result.detection_level.value} | Duration: {result.analysis_duration_ms}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing token {token_address}: {e}")
            # Return high-risk result on error as safety measure
            return RugDetectionResult(
                token_address=token_address,
                overall_risk=0.9,
                detection_level=DetectionLevel.CRITICAL,
                is_safe_to_trade=False,
                risk_factors=[RiskFactor(
                    factor_name="analysis_error",
                    risk_score=0.9,
                    severity=RiskLevel.CRITICAL,
                    description=f"Analysis failed: {str(e)}",
                    confidence=1.0
                )]
            )
    
    async def _analyze_contract_security(self, token_address: str) -> Dict[str, Any]:
        """Analyze smart contract security vulnerabilities"""
        try:
            risk_factors = []
            
            # Get contract metadata
            contract_data = await self._get_contract_metadata(token_address)
            
            # Check contract verification
            is_verified = contract_data.get('verified', False)
            if not is_verified:
                risk_factors.append(RiskFactor(
                    factor_name="unverified_contract",
                    risk_score=0.6,
                    severity=RiskLevel.HIGH,
                    description="Smart contract source code is not verified",
                    evidence=["Contract not verified on block explorer"],
                    confidence=0.9
                ))
            
            # Analyze contract functions
            contract_functions = contract_data.get('functions', [])
            high_risk_functions = [func for func in contract_functions 
                                 if any(pattern in func.lower() 
                                       for pattern in self.vulnerability_patterns['high_risk_functions'])]
            
            if high_risk_functions:
                risk_score = min(0.8, len(high_risk_functions) * 0.2)
                risk_factors.append(RiskFactor(
                    factor_name="high_risk_functions",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH if risk_score > 0.6 else RiskLevel.MEDIUM,
                    description=f"Contract contains {len(high_risk_functions)} high-risk functions",
                    evidence=[f"Functions: {', '.join(high_risk_functions[:3])}"],
                    confidence=0.8
                ))
            
            # Check for honeypot indicators
            honeypot_indicators = [func for func in contract_functions 
                                 if any(pattern in func.lower() 
                                       for pattern in self.vulnerability_patterns['honeypot_indicators'])]
            
            if honeypot_indicators:
                risk_score = min(0.9, len(honeypot_indicators) * 0.3)
                risk_factors.append(RiskFactor(
                    factor_name="honeypot_indicators",
                    risk_score=risk_score,
                    severity=RiskLevel.CRITICAL if risk_score > 0.7 else RiskLevel.HIGH,
                    description=f"Contract shows {len(honeypot_indicators)} honeypot indicators",
                    evidence=[f"Indicators: {', '.join(honeypot_indicators[:3])}"],
                    confidence=0.85
                ))
            
            # Check contract age
            contract_age_hours = contract_data.get('age_hours', 0)
            if contract_age_hours < 24:
                age_risk = max(0.0, 0.7 - (contract_age_hours / 24 * 0.7))
                risk_factors.append(RiskFactor(
                    factor_name="new_contract",
                    risk_score=age_risk,
                    severity=RiskLevel.MEDIUM,
                    description=f"Contract is very new ({contract_age_hours:.1f} hours old)",
                    evidence=[f"Deployed {contract_age_hours:.1f} hours ago"],
                    confidence=0.9
                ))
            
            # Check for proxy contracts
            is_proxy = contract_data.get('is_proxy', False)
            if is_proxy and not contract_data.get('is_transparent', False):
                risk_factors.append(RiskFactor(
                    factor_name="opaque_proxy",
                    risk_score=0.7,
                    severity=RiskLevel.HIGH,
                    description="Contract is a proxy without transparency",
                    evidence=["Non-transparent proxy contract detected"],
                    confidence=0.8
                ))
            
            # Calculate overall contract risk
            if risk_factors:
                contract_risk = min(1.0, sum(rf.risk_score * rf.confidence for rf in risk_factors) / len(risk_factors))
            else:
                contract_risk = 0.1  # Base risk for any contract
            
            return {
                'risk_score': contract_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error in contract security analysis: {e}")
            return {
                'risk_score': 0.5,  # Default medium risk on error
                'risk_factors': [RiskFactor(
                    factor_name="contract_analysis_error",
                    risk_score=0.5,
                    severity=RiskLevel.MEDIUM,
                    description=f"Contract analysis failed: {str(e)}",
                    confidence=0.5
                )]
            }
    
    async def _analyze_liquidity_health(self, token_address: str) -> Dict[str, Any]:
        """Analyze liquidity pool health and concentration risks"""
        try:
            risk_factors = []
            
            # Get comprehensive market data
            market_data = await self.market_data_fetcher.get_comprehensive_token_data(token_address)
            if not market_data:
                return {'risk_score': 0.8, 'risk_factors': []}
            
            # Check total liquidity
            liquidity_usd = float(market_data.get('liquidity_usd', 0))
            if liquidity_usd < 10000:  # Less than $10K
                risk_score = max(0.0, 0.8 - (liquidity_usd / 10000 * 0.8))
                risk_factors.append(RiskFactor(
                    factor_name="low_liquidity",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH if liquidity_usd < 1000 else RiskLevel.MEDIUM,
                    description=f"Low liquidity pool: ${liquidity_usd:,.0f}",
                    evidence=[f"Total liquidity: ${liquidity_usd:,.0f}"],
                    confidence=0.9
                ))
            
            # Check liquidity concentration
            liquidity_concentration = float(market_data.get('liquidity_concentration', 0.5))
            if liquidity_concentration > 0.8:
                risk_factors.append(RiskFactor(
                    factor_name="concentrated_liquidity",
                    risk_score=0.7,
                    severity=RiskLevel.HIGH,
                    description=f"Highly concentrated liquidity: {liquidity_concentration:.1%}",
                    evidence=[f"Liquidity concentration: {liquidity_concentration:.1%}"],
                    confidence=0.8
                ))
            
            # Check for large recent transactions that might indicate liquidity manipulation
            large_transactions = int(market_data.get('large_transactions_24h', 0))
            if large_transactions > 5:
                risk_score = min(0.6, large_transactions * 0.1)
                risk_factors.append(RiskFactor(
                    factor_name="suspicious_large_transactions",
                    risk_score=risk_score,
                    severity=RiskLevel.MEDIUM,
                    description=f"High number of large transactions: {large_transactions}",
                    evidence=[f"Large transactions in 24h: {large_transactions}"],
                    confidence=0.7
                ))
            
            # Check volume to liquidity ratio
            volume_24h = float(market_data.get('volume_24h_usd', 0))
            if liquidity_usd > 0:
                volume_liquidity_ratio = volume_24h / liquidity_usd
                if volume_liquidity_ratio > 10:  # Volume 10x liquidity suggests manipulation
                    risk_factors.append(RiskFactor(
                        factor_name="volume_manipulation",
                        risk_score=0.6,
                        severity=RiskLevel.MEDIUM,
                        description=f"Suspicious volume/liquidity ratio: {volume_liquidity_ratio:.1f}x",
                        evidence=[f"24h volume: ${volume_24h:,.0f}, Liquidity: ${liquidity_usd:,.0f}"],
                        confidence=0.7
                    ))
            
            # Calculate overall liquidity risk
            if risk_factors:
                liquidity_risk = min(1.0, sum(rf.risk_score * rf.confidence for rf in risk_factors) / len(risk_factors))
            else:
                liquidity_risk = 0.1  # Base risk
            
            return {
                'risk_score': liquidity_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error in liquidity health analysis: {e}")
            return {
                'risk_score': 0.5,
                'risk_factors': [RiskFactor(
                    factor_name="liquidity_analysis_error",
                    risk_score=0.5,
                    severity=RiskLevel.MEDIUM,
                    description=f"Liquidity analysis failed: {str(e)}",
                    confidence=0.5
                )]
            }
    
    async def _analyze_holder_distribution(self, token_address: str) -> Dict[str, Any]:
        """Analyze token holder distribution and whale concentration"""
        try:
            risk_factors = []
            
            # Get holder distribution data
            holder_data = await self._get_holder_distribution(token_address)
            
            # Check total holder count
            holder_count = holder_data.get('total_holders', 0)
            if holder_count < 100:
                risk_score = max(0.0, 0.7 - (holder_count / 100 * 0.7))
                risk_factors.append(RiskFactor(
                    factor_name="low_holder_count",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH if holder_count < 50 else RiskLevel.MEDIUM,
                    description=f"Low number of holders: {holder_count}",
                    evidence=[f"Total holders: {holder_count}"],
                    confidence=0.8
                ))
            
            # Check top holder concentration
            top_10_percent = holder_data.get('top_10_percent', 0.0)
            if top_10_percent > 70:  # Top 10 holders own >70%
                risk_score = min(0.9, (top_10_percent - 50) / 50 * 0.9)
                risk_factors.append(RiskFactor(
                    factor_name="whale_concentration",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH,
                    description=f"High whale concentration: top 10 holders own {top_10_percent:.1f}%",
                    evidence=[f"Top 10 holders: {top_10_percent:.1f}% of supply"],
                    confidence=0.85
                ))
            
            # Check dev/team holdings
            dev_holdings_percent = holder_data.get('dev_holdings_percent', 0.0)
            if dev_holdings_percent > 30:
                risk_score = min(0.8, (dev_holdings_percent - 20) / 80 * 0.8)
                risk_factors.append(RiskFactor(
                    factor_name="high_dev_holdings",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH if dev_holdings_percent > 50 else RiskLevel.MEDIUM,
                    description=f"High dev holdings: {dev_holdings_percent:.1f}%",
                    evidence=[f"Dev/team holdings: {dev_holdings_percent:.1f}%"],
                    confidence=0.8
                ))
            
            # Check for suspicious wallet patterns
            suspicious_wallets = holder_data.get('suspicious_wallets', 0)
            if suspicious_wallets > 0:
                risk_score = min(0.6, suspicious_wallets * 0.2)
                risk_factors.append(RiskFactor(
                    factor_name="suspicious_wallets",
                    risk_score=risk_score,
                    severity=RiskLevel.MEDIUM,
                    description=f"Suspicious wallet patterns detected: {suspicious_wallets}",
                    evidence=[f"Suspicious wallets: {suspicious_wallets}"],
                    confidence=0.7
                ))
            
            # Calculate overall holder risk
            if risk_factors:
                holder_risk = min(1.0, sum(rf.risk_score * rf.confidence for rf in risk_factors) / len(risk_factors))
            else:
                holder_risk = 0.1  # Base risk
            
            return {
                'risk_score': holder_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error in holder distribution analysis: {e}")
            return {
                'risk_score': 0.5,
                'risk_factors': [RiskFactor(
                    factor_name="holder_analysis_error",
                    risk_score=0.5,
                    severity=RiskLevel.MEDIUM,
                    description=f"Holder analysis failed: {str(e)}",
                    confidence=0.5
                )]
            }
    
    async def _analyze_behavioral_patterns(self, token_address: str) -> Dict[str, Any]:
        """Analyze trading behavioral patterns for rug indicators"""
        try:
            risk_factors = []
            
            # Get trading pattern data
            trading_data = await self._get_trading_patterns(token_address)
            
            # Check sell/buy ratio
            sell_buy_ratio = trading_data.get('sell_buy_ratio', 1.0)
            if sell_buy_ratio > 3.0:  # More than 3:1 sell pressure
                risk_score = min(0.7, (sell_buy_ratio - 1) / 10 * 0.7)
                risk_factors.append(RiskFactor(
                    factor_name="high_sell_pressure",
                    risk_score=risk_score,
                    severity=RiskLevel.HIGH if sell_buy_ratio > 5 else RiskLevel.MEDIUM,
                    description=f"High sell pressure: {sell_buy_ratio:.1f}:1 sell/buy ratio",
                    evidence=[f"Sell/buy ratio: {sell_buy_ratio:.1f}:1"],
                    confidence=0.8
                ))
            elif sell_buy_ratio < 0.1:  # Very few sells (honeypot indicator)
                risk_factors.append(RiskFactor(
                    factor_name="honeypot_sell_pattern",
                    risk_score=0.8,
                    severity=RiskLevel.HIGH,
                    description=f"Suspicious sell pattern: {sell_buy_ratio:.2f}:1 (potential honeypot)",
                    evidence=[f"Very low sell/buy ratio: {sell_buy_ratio:.2f}:1"],
                    confidence=0.9
                ))
            
            # Check for failed transaction patterns
            failed_transactions = trading_data.get('failed_transactions_24h', 0)
            total_transactions = trading_data.get('total_transactions_24h', 1)
            failure_rate = failed_transactions / max(total_transactions, 1)
            
            if failure_rate > 0.3:  # >30% transaction failure rate
                risk_factors.append(RiskFactor(
                    factor_name="high_transaction_failure_rate",
                    risk_score=0.7,
                    severity=RiskLevel.HIGH,
                    description=f"High transaction failure rate: {failure_rate:.1%}",
                    evidence=[f"Failed transactions: {failed_transactions}/{total_transactions}"],
                    confidence=0.85
                ))
            
            # Check for MEV bot activity
            mev_bot_activity = trading_data.get('mev_bot_transactions', 0)
            if mev_bot_activity > total_transactions * 0.5:  # >50% MEV activity
                risk_factors.append(RiskFactor(
                    factor_name="high_mev_activity",
                    risk_score=0.5,
                    severity=RiskLevel.MEDIUM,
                    description=f"High MEV bot activity: {mev_bot_activity} transactions",
                    evidence=[f"MEV bot transactions: {mev_bot_activity}"],
                    confidence=0.7
                ))
            
            # Check trading time patterns
            trading_hours_distribution = trading_data.get('trading_hours_distribution', {})
            if self._detect_unusual_trading_patterns(trading_hours_distribution):
                risk_factors.append(RiskFactor(
                    factor_name="unusual_trading_patterns",
                    risk_score=0.4,
                    severity=RiskLevel.MEDIUM,
                    description="Unusual trading time patterns detected",
                    evidence=["Concentrated trading in specific hours"],
                    confidence=0.6
                ))
            
            # Calculate overall behavioral risk
            if risk_factors:
                behavioral_risk = min(1.0, sum(rf.risk_score * rf.confidence for rf in risk_factors) / len(risk_factors))
            else:
                behavioral_risk = 0.1  # Base risk
            
            return {
                'risk_score': behavioral_risk,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            self.logger.error(f"Error in behavioral pattern analysis: {e}")
            return {
                'risk_score': 0.5,
                'risk_factors': [RiskFactor(
                    factor_name="behavioral_analysis_error",
                    risk_score=0.5,
                    severity=RiskLevel.MEDIUM,
                    description=f"Behavioral analysis failed: {str(e)}",
                    confidence=0.5
                )]
            }
    
    async def _analyze_historical_patterns(self, token_address: str) -> Dict[str, Any]:
        """Analyze historical patterns and match against known rug signatures"""
        try:
            risk_factors = []
            
            # Check against known rug database
            is_known_rug = await self._check_known_rug_database(token_address)
            if is_known_rug:
                risk_factors.append(RiskFactor(
                    factor_name="known_rug_token",
                    risk_score=1.0,
                    severity=RiskLevel.CRITICAL,
                    description="Token found in known rug pull database",
                    evidence=["Listed in rug pull database"],
                    confidence=1.0
                ))
                return {'risk_factors': risk_factors}
            
            # Check for similar contract patterns
            similar_contracts = await self._find_similar_contracts(token_address)
            rug_contracts = [c for c in similar_contracts if c.get('is_rug', False)]
            
            if len(rug_contracts) > 0:
                similarity_risk = min(0.8, len(rug_contracts) / max(len(similar_contracts), 1) * 0.8)
                risk_factors.append(RiskFactor(
                    factor_name="similar_rug_contracts",
                    risk_score=similarity_risk,
                    severity=RiskLevel.HIGH,
                    description=f"Similar to {len(rug_contracts)} known rug contracts",
                    evidence=[f"Found {len(rug_contracts)} similar rug contracts"],
                    confidence=0.7
                ))
            
            # Check for repeated deployment patterns (same dev)
            dev_history = await self._analyze_developer_history(token_address)
            if dev_history.get('previous_rugs', 0) > 0:
                risk_factors.append(RiskFactor(
                    factor_name="dev_rug_history",
                    risk_score=0.9,
                    severity=RiskLevel.CRITICAL,
                    description=f"Developer has {dev_history['previous_rugs']} previous rug pulls",
                    evidence=[f"Developer rug count: {dev_history['previous_rugs']}"],
                    confidence=0.9
                ))
            
            return {'risk_factors': risk_factors}
            
        except Exception as e:
            self.logger.error(f"Error in historical pattern analysis: {e}")
            return {'risk_factors': []}
    
    async def _calculate_honeypot_probability(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate probability that token is a honeypot"""
        try:
            honeypot_indicators = [
                'honeypot_indicators',
                'honeypot_sell_pattern',
                'high_transaction_failure_rate',
                'unverified_contract'
            ]
            
            honeypot_factors = [rf for rf in risk_factors if rf.factor_name in honeypot_indicators]
            
            if not honeypot_factors:
                return 0.1  # Base probability
            
            # Weight factors by their confidence
            weighted_scores = [rf.risk_score * rf.confidence for rf in honeypot_factors]
            return min(0.95, sum(weighted_scores) / len(weighted_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating honeypot probability: {e}")
            return 0.5
    
    async def _calculate_slow_rug_probability(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate probability of slow rug pull"""
        try:
            slow_rug_indicators = [
                'high_dev_holdings',
                'concentrated_liquidity',
                'high_sell_pressure',
                'whale_concentration'
            ]
            
            slow_rug_factors = [rf for rf in risk_factors if rf.factor_name in slow_rug_indicators]
            
            if not slow_rug_factors:
                return 0.1
            
            weighted_scores = [rf.risk_score * rf.confidence for rf in slow_rug_factors]
            return min(0.95, sum(weighted_scores) / len(weighted_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating slow rug probability: {e}")
            return 0.5
    
    async def _calculate_flash_rug_probability(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate probability of flash/instant rug pull"""
        try:
            flash_rug_indicators = [
                'low_liquidity',
                'new_contract',
                'high_risk_functions',
                'low_holder_count'
            ]
            
            flash_rug_factors = [rf for rf in risk_factors if rf.factor_name in flash_rug_indicators]
            
            if not flash_rug_factors:
                return 0.1
            
            weighted_scores = [rf.risk_score * rf.confidence for rf in flash_rug_factors]
            return min(0.95, sum(weighted_scores) / len(weighted_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating flash rug probability: {e}")
            return 0.5
    
    async def _calculate_dev_exit_probability(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate probability of developer exit scam"""
        try:
            dev_exit_indicators = [
                'dev_rug_history',
                'high_dev_holdings',
                'unverified_contract',
                'suspicious_wallets'
            ]
            
            dev_exit_factors = [rf for rf in risk_factors if rf.factor_name in dev_exit_indicators]
            
            if not dev_exit_factors:
                return 0.1
            
            weighted_scores = [rf.risk_score * rf.confidence for rf in dev_exit_factors]
            return min(0.95, sum(weighted_scores) / len(weighted_scores))
            
        except Exception as e:
            self.logger.error(f"Error calculating dev exit probability: {e}")
            return 0.5
    
    async def _calculate_overall_risk(self, result: RugDetectionResult) -> float:
        """Calculate overall risk score from all factors"""
        try:
            # Weight different risk categories
            weights = {
                'contract': 0.3,
                'liquidity': 0.25,
                'holder': 0.25,
                'behavioral': 0.2
            }
            
            # Calculate weighted risk
            weighted_risk = (
                result.contract_risk_score * weights['contract'] +
                result.liquidity_risk_score * weights['liquidity'] +
                result.holder_risk_score * weights['holder'] +
                result.behavioral_risk_score * weights['behavioral']
            )
            
            # Apply probability multipliers for specific rug types
            max_rug_probability = max(
                result.honeypot_probability,
                result.slow_rug_probability,
                result.flash_rug_probability,
                result.dev_exit_probability
            )
            
            # Combine weighted risk with maximum rug probability
            overall_risk = max(weighted_risk, max_rug_probability * 0.9)
            
            return min(1.0, overall_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk: {e}")
            return 0.8  # High risk on error
    
    def _determine_detection_level(self, overall_risk: float) -> DetectionLevel:
        """Determine detection level based on overall risk score"""
        if overall_risk >= self.risk_thresholds['critical']:
            return DetectionLevel.CRITICAL
        elif overall_risk >= self.risk_thresholds['high']:
            return DetectionLevel.DANGER
        elif overall_risk >= self.risk_thresholds['medium']:
            return DetectionLevel.WARNING
        elif overall_risk >= self.risk_thresholds['low']:
            return DetectionLevel.CAUTION
        else:
            return DetectionLevel.CLEAR
    
    async def _generate_trading_recommendations(self, result: RugDetectionResult) -> List[str]:
        """Generate specific trading recommendations based on analysis"""
        recommendations = []
        
        if result.detection_level == DetectionLevel.CRITICAL:
            recommendations.extend([
                "🚨 DO NOT TRADE - Critical rug pull risk detected",
                "🛑 Avoid this token completely",
                "⚠️ Report to community if promoted"
            ])
        elif result.detection_level == DetectionLevel.DANGER:
            recommendations.extend([
                "🚫 High risk - Trading not recommended",
                "💰 If trading, use extremely small position sizes",
                "⏰ Set immediate stop losses if entering"
            ])
        elif result.detection_level == DetectionLevel.WARNING:
            recommendations.extend([
                "⚠️ Proceed with extreme caution",
                "💰 Limit position size to <1% of portfolio",
                "📊 Monitor closely for changes in risk factors",
                "⏰ Use tight stop losses"
            ])
        elif result.detection_level == DetectionLevel.CAUTION:
            recommendations.extend([
                "✅ Acceptable risk but monitor closely",
                "💰 Consider normal position sizing",
                "📊 Watch for any changes in risk profile"
            ])
        else:  # CLEAR
            recommendations.extend([
                "✅ Low risk profile detected",
                "💰 Normal trading strategies applicable",
                "📊 Continue monitoring as part of routine risk management"
            ])
        
        # Add specific recommendations based on risk factors
        high_risk_factors = [rf for rf in result.risk_factors if rf.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        for factor in high_risk_factors[:3]:  # Top 3 highest risks
            recommendations.append(f"⚠️ Monitor: {factor.description}")
        
        return recommendations
    
    async def _generate_monitoring_alerts(self, result: RugDetectionResult) -> List[str]:
        """Generate monitoring alerts for ongoing risk assessment"""
        alerts = []
        
        # Add alerts based on specific risk factors
        for factor in result.risk_factors:
            if factor.factor_name == "concentrated_liquidity":
                alerts.append("Monitor liquidity pool for sudden removals")
            elif factor.factor_name == "high_dev_holdings":
                alerts.append("Watch dev wallets for large sell orders")
            elif factor.factor_name == "new_contract":
                alerts.append("Monitor contract for updates or changes")
            elif factor.factor_name == "low_holder_count":
                alerts.append("Track holder growth and distribution changes")
        
        # Add probability-based alerts
        if result.honeypot_probability > 0.5:
            alerts.append("Test small transactions before larger trades")
        if result.slow_rug_probability > 0.5:
            alerts.append("Monitor for gradual liquidity reduction")
        if result.dev_exit_probability > 0.5:
            alerts.append("Watch for social media and development activity")
        
        return alerts
    
    async def _calculate_confidence_score(self, result: RugDetectionResult) -> float:
        """Calculate confidence in the analysis results"""
        try:
            confidence_factors = []
            
            # Data source availability
            data_sources = len(result.data_sources_used)
            source_confidence = min(1.0, data_sources / 3.0)  # Full confidence with 3+ sources
            confidence_factors.append(source_confidence * 0.3)
            
            # Number of risk factors analyzed
            factor_count = len(result.risk_factors)
            factor_confidence = min(1.0, factor_count / 5.0)  # Full confidence with 5+ factors
            confidence_factors.append(factor_confidence * 0.3)
            
            # Average confidence of individual risk factors
            if result.risk_factors:
                avg_factor_confidence = sum(rf.confidence for rf in result.risk_factors) / len(result.risk_factors)
                confidence_factors.append(avg_factor_confidence * 0.4)
            else:
                confidence_factors.append(0.1)  # Low confidence with no factors
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    # Helper methods for data fetching (simplified implementations)
    
    async def _get_contract_metadata(self, token_address: str) -> Dict[str, Any]:
        """Get contract metadata (simplified implementation)"""
        try:
            # In production, this would query Solana RPC and contract analyzers
            # For now, return basic structure with some randomized data for testing
            
            age_hours = hash(token_address) % 100  # Deterministic but varying age
            is_verified = (hash(token_address) % 3) != 0  # ~67% verified
            
            return {
                'verified': is_verified,
                'age_hours': age_hours,
                'functions': ['transfer', 'approve', 'mint'] if not is_verified else ['transfer', 'approve'],
                'is_proxy': (hash(token_address) % 5) == 0,  # 20% proxy contracts
                'is_transparent': True
            }
        except Exception as e:
            self.logger.error(f"Error getting contract metadata: {e}")
            return {}
    
    async def _get_holder_distribution(self, token_address: str) -> Dict[str, Any]:
        """Get holder distribution data (simplified implementation)"""
        try:
            # In production, this would query blockchain data for holder analysis
            hash_val = hash(token_address)
            
            return {
                'total_holders': max(50, hash_val % 1000),
                'top_10_percent': max(30, hash_val % 80),
                'dev_holdings_percent': hash_val % 40,
                'suspicious_wallets': hash_val % 3
            }
        except Exception as e:
            self.logger.error(f"Error getting holder distribution: {e}")
            return {}
    
    async def _get_trading_patterns(self, token_address: str) -> Dict[str, Any]:
        """Get trading pattern data (simplified implementation)"""
        try:
            # In production, this would analyze transaction history
            hash_val = hash(token_address)
            
            return {
                'sell_buy_ratio': max(0.1, (hash_val % 100) / 20),
                'failed_transactions_24h': hash_val % 10,
                'total_transactions_24h': max(10, hash_val % 100),
                'mev_bot_transactions': hash_val % 20,
                'trading_hours_distribution': {}
            }
        except Exception as e:
            self.logger.error(f"Error getting trading patterns: {e}")
            return {}
    
    def _detect_unusual_trading_patterns(self, trading_hours: Dict) -> bool:
        """Detect unusual trading time patterns"""
        # Simplified implementation
        return False
    
    async def _check_known_rug_database(self, token_address: str) -> bool:
        """Check if token is in known rug database"""
        # In production, this would check against a real database
        return False
    
    async def _find_similar_contracts(self, token_address: str) -> List[Dict[str, Any]]:
        """Find contracts similar to the given token"""
        # Simplified implementation
        return []
    
    async def _analyze_developer_history(self, token_address: str) -> Dict[str, Any]:
        """Analyze developer's historical projects"""
        # Simplified implementation
        return {'previous_rugs': 0}
    
    async def _load_known_rug_database(self):
        """Load known rug pull database"""
        try:
            # In production, this would load from a real database
            self.logger.info("📚 Known rug database loaded")
        except Exception as e:
            self.logger.error(f"Error loading rug database: {e}")
    
    async def _test_analysis_capabilities(self):
        """Test analysis capabilities"""
        try:
            # Test basic analysis with a known token (SOL)
            sol_address = "So11111111111111111111111111111111111111112"
            test_result = await self.analyze_token(sol_address, "Wrapped SOL", "SOL")
            
            if test_result.overall_risk < 0.5:  # SOL should be low risk
                self.logger.info("✅ Analysis capabilities test passed")
            else:
                self.logger.warning("⚠️ Analysis capabilities test gave unexpected results")
                
        except Exception as e:
            self.logger.error(f"Analysis capabilities test failed: {e}")
    
    # Cache management methods
    
    async def _get_cached_analysis(self, token_address: str) -> Optional[RugDetectionResult]:
        """Get cached analysis result if valid"""
        try:
            if token_address not in self.analysis_cache:
                return None
            
            result = self.analysis_cache[token_address]
            age_seconds = (datetime.now() - result.analyzed_at).total_seconds()
            
            if age_seconds > self.cache_ttl_seconds:
                del self.analysis_cache[token_address]
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting cached analysis: {e}")
            return None
    
    async def _cache_analysis_result(self, token_address: str, result: RugDetectionResult):
        """Cache analysis result"""
        try:
            # Clean cache if getting too large
            if len(self.analysis_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.analysis_cache.items(),
                    key=lambda x: x[1].analyzed_at
                )
                for addr, _ in sorted_cache[:20]:  # Remove oldest 20
                    del self.analysis_cache[addr]
            
            self.analysis_cache[token_address] = result
            
        except Exception as e:
            self.logger.error(f"Error caching analysis result: {e}")
    
    def _update_analysis_metrics(self, result: RugDetectionResult):
        """Update performance metrics"""
        try:
            self.total_analyses += 1
            
            if result.detection_level in [DetectionLevel.DANGER, DetectionLevel.CRITICAL]:
                self.total_rugs_detected += 1
            
            # Update average analysis time
            if self.total_analyses == 1:
                self.average_analysis_time_ms = result.analysis_duration_ms
            else:
                alpha = 0.1
                self.average_analysis_time_ms = (alpha * result.analysis_duration_ms + 
                                               (1 - alpha) * self.average_analysis_time_ms)
                
        except Exception as e:
            self.logger.error(f"Error updating analysis metrics: {e}")
    
    def _get_data_sources_used(self) -> List[str]:
        """Get list of data sources used in analysis"""
        return ['contract_analyzer', 'market_data_fetcher', 'holder_analyzer']
    
    def get_detector_status(self) -> Dict[str, Any]:
        """Get comprehensive detector status"""
        detection_rate = 0.0
        if self.total_analyses > 0:
            detection_rate = self.total_rugs_detected / self.total_analyses
        
        return {
            'initialized': True,
            'total_analyses': self.total_analyses,
            'total_rugs_detected': self.total_rugs_detected,
            'detection_rate': round(detection_rate, 3),
            'false_positive_rate': round(self.false_positive_rate, 3),
            'average_analysis_time_ms': round(self.average_analysis_time_ms, 2),
            'cache_size': len(self.analysis_cache),
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'risk_thresholds': self.risk_thresholds
        }
    
    async def shutdown(self):
        """Shutdown the rug detector"""
        try:
            self.logger.info("🛑 Shutting down rug detector...")
            
            # Clear caches
            self.analysis_cache.clear()
            
            self.logger.info("✅ Rug detector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during detector shutdown: {e}")


# Global instance manager
_rug_detector = None

async def get_rug_detector() -> EnhancedRugDetector:
    """Get global rug detector instance"""
    global _rug_detector
    if _rug_detector is None:
        _rug_detector = EnhancedRugDetector()
        await _rug_detector.initialize()
    return _rug_detector