"""
ENHANCED RUG DETECTOR - BATTLEFIELD INTELLIGENCE
=============================================

Advanced rug detection system with real-time monitoring,
pattern recognition, and multi-factor analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

from worker_ant_v1.core.unified_config import get_security_config
from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.trading.analyzers import (
    LiquidityAnalyzer,
    OwnershipAnalyzer,
    CodeAnalyzer,
    TradingAnalyzer
)

rug_logger = setup_logger(__name__)


class RugDetectionLevel(Enum):
    """Rug detection severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RugDetectionResult:
    """Result of rug detection analysis"""
    level: RugDetectionLevel
    confidence: float
    reasons: List[str]
    timestamp: datetime
    token_address: str
    token_name: str
    token_symbol: str
    # Risk metrics
    liquidity_risk: float
    ownership_risk: float
    code_risk: float
    trading_risk: float
    # Detailed analysis
    liquidity_analysis: Dict[str, float]
    ownership_analysis: Dict[str, float]
    code_analysis: Dict[str, Dict]
    trading_analysis: Dict[str, float]


class EnhancedRugDetector:
    """Advanced rug detection with multi-factor analysis"""
    def __init__(self):
        self.config = get_security_config()
        self.logger = rug_logger
        # Risk thresholds
        self.liquidity_threshold = self.config.get(
            "liquidity_threshold", 0.7
        )
        self.ownership_threshold = self.config.get(
            "ownership_threshold", 0.8
        )
        self.code_threshold = self.config.get(
            "code_threshold", 0.6
        )
        self.trading_threshold = self.config.get(
            "trading_threshold", 0.75
        )
        # Analysis weights
        self.liquidity_weight = 0.3
        self.ownership_weight = 0.3
        self.code_weight = 0.2
        self.trading_weight = 0.2
        # Initialize analysis components
        self._init_analyzers()

    def _init_analyzers(self):
        """Initialize analysis components"""
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.ownership_analyzer = OwnershipAnalyzer()
        self.code_analyzer = CodeAnalyzer()
        self.trading_analyzer = TradingAnalyzer()

    async def analyze_token(
        self,
        token_address: str,
        token_name: str,
        token_symbol: str
    ) -> RugDetectionResult:
        """
        Analyze token for potential rug pull indicators.
        Returns comprehensive analysis results.
        """
        try:
            # Run analysis components in parallel
            liquidity_result = await self.liquidity_analyzer.analyze(
                token_address
            )
            ownership_result = await self.ownership_analyzer.analyze(
                token_address
            )
            code_result = await self.code_analyzer.analyze(
                token_address
            )
            trading_result = await self.trading_analyzer.analyze(
                token_address
            )
            # Calculate risk scores
            liquidity_risk = self._calculate_liquidity_risk(
                liquidity_result
            )
            ownership_risk = self._calculate_ownership_risk(
                ownership_result
            )
            code_risk = self._calculate_code_risk(code_result)
            trading_risk = self._calculate_trading_risk(trading_result)
            # Calculate overall risk and confidence
            overall_risk = (
                liquidity_risk * self.liquidity_weight +
                ownership_risk * self.ownership_weight +
                code_risk * self.code_weight +
                trading_risk * self.trading_weight
            )
            confidence = self._calculate_confidence([
                liquidity_result,
                ownership_result,
                code_result,
                trading_result
            ])
            # Determine detection level
            level = self._determine_detection_level(
                overall_risk,
                confidence
            )
            # Compile reasons
            reasons = self._compile_detection_reasons(
                liquidity_result,
                ownership_result,
                code_result,
                trading_result
            )
            return RugDetectionResult(
                level=level,
                confidence=confidence,
                reasons=reasons,
                timestamp=datetime.utcnow(),
                token_address=token_address,
                token_name=token_name,
                token_symbol=token_symbol,
                liquidity_risk=liquidity_risk,
                ownership_risk=ownership_risk,
                code_risk=code_risk,
                trading_risk=trading_risk,
                liquidity_analysis=liquidity_result,
                ownership_analysis=ownership_result,
                code_analysis=code_result,
                trading_analysis=trading_result
            )
        except Exception as e:
            self.logger.error(
                f"Error analyzing token {token_address}: {str(e)}"
            )
            raise
    
    async def analyze_token_simple(self, token_address: str, market_data: Dict[str, Any] = None) -> float:
        """Analyze token for rug detection - simplified interface for testing"""
        try:
            # Extract token info from market data if available
            token_name = market_data.get('name', 'Unknown') if market_data else 'Unknown'
            token_symbol = market_data.get('symbol', 'UNK') if market_data else 'UNK'
            
            # Run full analysis
            result = await self.analyze_token(token_address, token_name, token_symbol)
            
            # Return risk score as float (0.0 = safe, 1.0 = high risk)
            risk_mapping = {
                RugDetectionLevel.LOW: 0.1,
                RugDetectionLevel.MEDIUM: 0.4,
                RugDetectionLevel.HIGH: 0.7,
                RugDetectionLevel.CRITICAL: 0.9
            }
            
            return risk_mapping.get(result.level, 0.5)
            
        except Exception as e:
            self.logger.error(f"Error in simplified analyze_token: {e}")
            return 0.5  # Default to medium risk on error
    
    async def analyze_token(self, token_address: str, market_data: Dict[str, Any] = None) -> float:
        """Analyze token for rug detection - overloaded method for testing compatibility"""
        return await self.analyze_token_simple(token_address, market_data)

    def _calculate_liquidity_risk(
        self,
        liquidity_result: Dict[str, float]
    ) -> float:
        """Calculate liquidity-based risk score"""
        risk_score = 0.0
        # Weight different liquidity factors
        if liquidity_result["total_liquidity"] < 1000:
            risk_score += 0.4
        if liquidity_result["liquidity_concentration"] > 0.8:
            risk_score += 0.3
        if liquidity_result["liquidity_age_hours"] < 24:
            risk_score += 0.3
        return min(risk_score, 1.0)

    def _calculate_ownership_risk(
        self,
        ownership_result: Dict[str, float]
    ) -> float:
        """Calculate ownership-based risk score"""
        risk_score = 0.0
        # Weight different ownership factors
        if ownership_result["top_holder_percentage"] > 0.5:
            risk_score += 0.4
        if ownership_result["contract_ownership"] == "centralized":
            risk_score += 0.3
        if ownership_result["holder_concentration"] > 0.7:
            risk_score += 0.3
        return min(risk_score, 1.0)

    def _calculate_code_risk(
        self,
        code_result: Dict[str, Dict]
    ) -> float:
        """Calculate code-based risk score"""
        risk_score = 0.0
        # Weight different code factors
        if code_result["has_honeypot_code"]:
            risk_score += 0.5
        if code_result["has_backdoor"]:
            risk_score += 0.3
        if code_result["is_proxy"] and not code_result["is_transparent"]:
            risk_score += 0.2
        return min(risk_score, 1.0)

    def _calculate_trading_risk(
        self,
        trading_result: Dict[str, float]
    ) -> float:
        """Calculate trading pattern-based risk score"""
        risk_score = 0.0
        # Weight different trading factors
        if trading_result["buy_tax"] > 0.1:
            risk_score += 0.3
        if trading_result["sell_tax"] > 0.1:
            risk_score += 0.3
        if trading_result["price_manipulation_score"] > 0.7:
            risk_score += 0.4
        return min(risk_score, 1.0)

    def _calculate_confidence(
        self,
        analysis_results: List[Dict]
    ) -> float:
        """Calculate confidence score for the analysis"""
        # Base confidence on data availability and consistency
        confidence = 1.0
        for result in analysis_results:
            if not result or len(result) < 3:
                confidence *= 0.8
            if any(v is None for v in result.values()):
                confidence *= 0.9
        return max(min(confidence, 1.0), 0.1)

    def _determine_detection_level(
        self,
        risk_score: float,
        confidence: float
    ) -> RugDetectionLevel:
        """Determine rug detection level based on risk and confidence"""
        if risk_score > 0.8 and confidence > 0.7:
            return RugDetectionLevel.CRITICAL
        elif risk_score > 0.6 and confidence > 0.6:
            return RugDetectionLevel.HIGH
        elif risk_score > 0.4 and confidence > 0.5:
            return RugDetectionLevel.MEDIUM
        else:
            return RugDetectionLevel.LOW

    def _compile_detection_reasons(
        self,
        liquidity_result: Dict[str, float],
        ownership_result: Dict[str, float],
        code_result: Dict[str, Dict],
        trading_result: Dict[str, float]
    ) -> List[str]:
        """Compile list of detection reasons"""
        reasons = []
        # Add liquidity-related reasons
        if liquidity_result["total_liquidity"] < 1000:
            reasons.append("Low total liquidity")
        if liquidity_result["liquidity_concentration"] > 0.8:
            reasons.append("High liquidity concentration")
        # Add ownership-related reasons
        if ownership_result["top_holder_percentage"] > 0.5:
            reasons.append("High token concentration")
        if ownership_result["contract_ownership"] == "centralized":
            reasons.append("Centralized contract ownership")
        # Add code-related reasons
        if code_result["has_honeypot_code"]:
            reasons.append("Potential honeypot code detected")
        if code_result["has_backdoor"]:
            reasons.append("Potential backdoor detected")
        # Add trading-related reasons
        if trading_result["buy_tax"] > 0.1:
            reasons.append(f"High buy tax: {trading_result['buy_tax']*100}%")
        if trading_result["sell_tax"] > 0.1:
            reasons.append(f"High sell tax: {trading_result['sell_tax']*100}%")
        return reasons