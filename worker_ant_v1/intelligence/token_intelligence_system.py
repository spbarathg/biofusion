"""
TOKEN INTELLIGENCE SYSTEM - COMPREHENSIVE TOKEN ANALYSIS
======================================================

Advanced token analysis system that evaluates tokens for trading opportunities
using multiple data sources and AI-powered pattern recognition.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import aiohttp

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.intelligence.sentiment_analyzer import SentimentAnalyzer
from worker_ant_v1.intelligence.technical_analyzer import TechnicalAnalyzer
from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector

class TokenCategory(Enum):
    """Token categories based on analysis"""
    GEM = "gem"                    # High potential token
    PUMP = "pump"                  # Pump and dump token
    RUG = "rug"                    # Rug pull risk
    STABLE = "stable"              # Stable, low volatility
    VOLATILE = "volatile"          # High volatility
    UNKNOWN = "unknown"            # Insufficient data

class OpportunityLevel(Enum):
    """Trading opportunity levels"""
    HIGH = "high"                  # Strong buy signal
    MEDIUM = "medium"              # Moderate opportunity
    LOW = "low"                    # Weak opportunity
    AVOID = "avoid"                # Avoid this token
    MONITOR = "monitor"            # Monitor for changes

@dataclass
class TokenMetrics:
    """Comprehensive token metrics"""
    token_address: str
    symbol: str
    price: float
    market_cap: float
    volume_24h: float
    liquidity: float
    holder_count: int
    age_hours: float
    price_change_1h: float
    price_change_24h: float
    volume_change_24h: float
    timestamp: datetime

@dataclass
class TokenAnalysis:
    """Complete token analysis result"""
    token_address: str
    token_metrics: TokenMetrics
    category: TokenCategory
    opportunity_level: OpportunityLevel
    confidence_score: float
    risk_score: float
    sentiment_score: float
    technical_score: float
    security_score: float
    pattern_signals: List[str]
    risk_indicators: List[str]
    recommendation: str
    reasoning: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class TokenIntelligenceSystem:
    """Advanced token intelligence and analysis system"""
    
    def __init__(self):
        self.logger = setup_logger("TokenIntelligenceSystem")
        
        # Analysis components
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.technical_analyzer: Optional[TechnicalAnalyzer] = None
        self.rug_detector: Optional[EnhancedRugDetector] = None
        
        # Data storage
        self.token_cache: Dict[str, TokenAnalysis] = {}
        self.analysis_history: Dict[str, List[TokenAnalysis]] = {}
        self.pattern_database: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.config = {
            'cache_ttl_minutes': 15,
            'min_liquidity_threshold': 1000.0,  # SOL
            'min_holder_count': 50,
            'max_analysis_age_hours': 24,
            'pattern_recognition_enabled': True,
            'sentiment_weight': 0.3,
            'technical_weight': 0.4,
            'security_weight': 0.3
        }
        
        # System state
        self.initialized = False
        self.analysis_active = False
        
    async def initialize(self) -> bool:
        """Initialize the token intelligence system"""
        try:
            self.logger.info("🧠 Initializing Token Intelligence System...")
            
            # Initialize analysis components
            self.sentiment_analyzer = SentimentAnalyzer()
            self.technical_analyzer = TechnicalAnalyzer()
            self.rug_detector = EnhancedRugDetector()
            
            # Initialize components
            await self.rug_detector.initialize()
            
            # Start background tasks
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._pattern_analysis_loop())
            
            self.initialized = True
            self.logger.info("✅ Token Intelligence System initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize token intelligence: {e}")
            return False
    
    async def analyze_token(self, token_address: str, market_data: Dict[str, Any] = None) -> TokenAnalysis:
        """Analyze a token comprehensively"""
        try:
            # Check cache first
            cached_analysis = self._get_cached_analysis(token_address)
            if cached_analysis:
                return cached_analysis
            
            # Get token metrics
            token_metrics = await self._get_token_metrics(token_address, market_data)
            
            # Perform comprehensive analysis
            analysis = await self._perform_comprehensive_analysis(token_address, token_metrics)
            
            # Cache the analysis
            self._cache_analysis(token_address, analysis)
            
            # Store in history
            if token_address not in self.analysis_history:
                self.analysis_history[token_address] = []
            self.analysis_history[token_address].append(analysis)
            
            # Keep only last 50 analyses per token
            if len(self.analysis_history[token_address]) > 50:
                self.analysis_history[token_address] = self.analysis_history[token_address][-50:]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing token {token_address}: {e}")
            return self._create_fallback_analysis(token_address)
    
    async def _get_token_metrics(self, token_address: str, market_data: Dict[str, Any] = None) -> TokenMetrics:
        """Get comprehensive token metrics"""
        try:
            if market_data:
                # Use provided market data
                metrics = TokenMetrics(
                    token_address=token_address,
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    price=market_data.get('price', 0.0),
                    market_cap=market_data.get('market_cap', 0.0),
                    volume_24h=market_data.get('volume_24h', 0.0),
                    liquidity=market_data.get('liquidity', 0.0),
                    holder_count=market_data.get('holder_count', 0),
                    age_hours=market_data.get('age_hours', 0.0),
                    price_change_1h=market_data.get('price_change_1h', 0.0),
                    price_change_24h=market_data.get('price_change_24h', 0.0),
                    volume_change_24h=market_data.get('volume_change_24h', 0.0),
                    timestamp=datetime.now()
                )
            else:
                # Fetch from APIs (placeholder implementation)
                metrics = await self._fetch_token_metrics(token_address)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting token metrics: {e}")
            return TokenMetrics(
                token_address=token_address,
                symbol="UNKNOWN",
                price=0.0,
                market_cap=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                holder_count=0,
                age_hours=0.0,
                price_change_1h=0.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=datetime.now()
            )
    
    async def _fetch_token_metrics(self, token_address: str) -> TokenMetrics:
        """Fetch token metrics from Jupiter and Birdeye APIs"""
        try:
            # Example: Fetch price and liquidity from Jupiter
            async with aiohttp.ClientSession() as session:
                # Jupiter price endpoint
                jup_url = f'https://price.jup.ag/v4/price?ids={token_address}'
                async with session.get(jup_url) as resp:
                    jup_data = await resp.json()
                price = jup_data.get('data', {}).get(token_address, {}).get('price', 0.0)
                # Birdeye liquidity endpoint
                be_url = f'https://public-api.birdeye.so/public/token/{token_address}/liquidity'
                async with session.get(be_url) as resp:
                    be_data = await resp.json()
                liquidity = be_data.get('data', {}).get('liquidity', 0.0)
            return TokenMetrics(
                token_address=token_address,
                symbol="UNKNOWN",
                price=price,
                market_cap=0.0,  # Add more API calls for mcap if needed
                volume_24h=0.0,  # Add more API calls for volume if needed
                liquidity=liquidity,
                holder_count=0,
                age_hours=0.0,
                price_change_1h=0.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error fetching token metrics: {e}")
            return TokenMetrics(
                token_address=token_address,
                symbol="UNKNOWN",
                price=0.0,
                market_cap=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                holder_count=0,
                age_hours=0.0,
                price_change_1h=0.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=datetime.now()
            )
    
    async def _perform_comprehensive_analysis(self, token_address: str, metrics: TokenMetrics) -> TokenAnalysis:
        """Perform comprehensive token analysis"""
        try:
            # Analyze sentiment
            sentiment_score = await self._analyze_sentiment(token_address, metrics)
            
            # Analyze technical indicators
            technical_score = await self._analyze_technical(token_address, metrics)
            
            # Analyze security
            security_score = await self._analyze_security(token_address, metrics)
            
            # Detect patterns
            pattern_signals = await self._detect_patterns(token_address, metrics)
            
            # Assess risks
            risk_indicators = await self._assess_risks(token_address, metrics)
            
            # Calculate composite scores
            confidence_score = self._calculate_confidence_score(sentiment_score, technical_score, security_score)
            risk_score = self._calculate_risk_score(risk_indicators, metrics)
            
            # Determine category and opportunity level
            category = self._determine_token_category(metrics, risk_score, confidence_score)
            opportunity_level = self._determine_opportunity_level(confidence_score, risk_score, pattern_signals)
            
            # Generate recommendation
            recommendation, reasoning = self._generate_recommendation(
                opportunity_level, category, confidence_score, risk_score, pattern_signals, risk_indicators
            )
            
            analysis = TokenAnalysis(
                token_address=token_address,
                token_metrics=metrics,
                category=category,
                opportunity_level=opportunity_level,
                confidence_score=confidence_score,
                risk_score=risk_score,
                sentiment_score=sentiment_score,
                technical_score=technical_score,
                security_score=security_score,
                pattern_signals=pattern_signals,
                risk_indicators=risk_indicators,
                recommendation=recommendation,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error performing comprehensive analysis: {e}")
            return self._create_fallback_analysis(token_address)
    
    async def _analyze_sentiment(self, token_address: str, metrics: TokenMetrics) -> float:
        """Analyze token sentiment"""
        try:
            if not self.sentiment_analyzer:
                return 0.5
            
            # Get sentiment data
            sentiment_data = await self.sentiment_analyzer.analyze_token_sentiment(token_address, {
                'symbol': metrics.symbol,
                'price': metrics.price,
                'volume': metrics.volume_24h
            })
            
            return sentiment_data.overall_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 0.5
    
    async def _analyze_technical(self, token_address: str, metrics: TokenMetrics) -> float:
        """Analyze technical indicators"""
        try:
            if not self.technical_analyzer:
                return 0.5
            
            # Get technical analysis
            technical_data = await self.technical_analyzer.analyze_token(token_address, {
                'price': metrics.price,
                'volume': metrics.volume_24h,
                'price_change_1h': metrics.price_change_1h,
                'price_change_24h': metrics.price_change_24h
            })
            
            return technical_data.get('overall_score', 0.5)
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical indicators: {e}")
            return 0.5
    
    async def _analyze_security(self, token_address: str, metrics: TokenMetrics) -> float:
        """Analyze token security"""
        try:
            if not self.rug_detector:
                return 0.5
            
            # Get security assessment
            security_data = await self.rug_detector.analyze_security(token_address)
            
            # Convert threat level to score
            threat_level = security_data.threat_level.value if hasattr(security_data.threat_level, 'value') else 0.5
            security_score = 1.0 - threat_level
            
            return security_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing security: {e}")
            return 0.5
    
    async def _detect_patterns(self, token_address: str, metrics: TokenMetrics) -> List[str]:
        """Detect trading patterns"""
        try:
            patterns = []
            
            # Volume spike pattern
            if metrics.volume_change_24h > 2.0:  # 200% volume increase
                patterns.append("volume_spike")
            
            # Price momentum pattern
            if metrics.price_change_1h > 0.1:  # 10% hourly gain
                patterns.append("price_momentum")
            
            # Liquidity pattern
            if metrics.liquidity > self.config['min_liquidity_threshold']:
                patterns.append("high_liquidity")
            else:
                patterns.append("low_liquidity")
            
            # Holder pattern
            if metrics.holder_count > self.config['min_holder_count']:
                patterns.append("decent_holders")
            else:
                patterns.append("few_holders")
            
            # Age pattern
            if metrics.age_hours < 24:
                patterns.append("new_token")
            elif metrics.age_hours < 168:  # 1 week
                patterns.append("recent_token")
            else:
                patterns.append("established_token")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _assess_risks(self, token_address: str, metrics: TokenMetrics) -> List[str]:
        """Assess token risks"""
        try:
            risks = []
            
            # Liquidity risk
            if metrics.liquidity < self.config['min_liquidity_threshold']:
                risks.append("low_liquidity")
            
            # Holder concentration risk
            if metrics.holder_count < self.config['min_holder_count']:
                risks.append("holder_concentration")
            
            # Age risk
            if metrics.age_hours < 1:
                risks.append("very_new_token")
            
            # Volume risk
            if metrics.volume_24h < 100:  # Less than 100 SOL volume
                risks.append("low_volume")
            
            # Price volatility risk
            if abs(metrics.price_change_1h) > 0.5:  # 50% hourly change
                risks.append("high_volatility")
            
            # Market cap risk
            if metrics.market_cap < 1000:  # Less than $1000 market cap
                risks.append("micro_cap")
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error assessing risks: {e}")
            return []
    
    def _calculate_confidence_score(self, sentiment_score: float, technical_score: float, security_score: float) -> float:
        """Calculate composite confidence score"""
        try:
            confidence = (
                sentiment_score * self.config['sentiment_weight'] +
                technical_score * self.config['technical_weight'] +
                security_score * self.config['security_weight']
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_risk_score(self, risk_indicators: List[str], metrics: TokenMetrics) -> float:
        """Calculate composite risk score"""
        try:
            base_risk = 0.5
            
            # Add risk for each indicator
            risk_multipliers = {
                'low_liquidity': 0.2,
                'holder_concentration': 0.15,
                'very_new_token': 0.1,
                'low_volume': 0.1,
                'high_volatility': 0.15,
                'micro_cap': 0.1
            }
            
            for indicator in risk_indicators:
                if indicator in risk_multipliers:
                    base_risk += risk_multipliers[indicator]
            
            return min(1.0, base_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _determine_token_category(self, metrics: TokenMetrics, risk_score: float, confidence_score: float) -> TokenCategory:
        """Determine token category"""
        try:
            # High risk = likely rug
            if risk_score > 0.8:
                return TokenCategory.RUG
            
            # High confidence + good metrics = gem
            if confidence_score > 0.7 and metrics.liquidity > self.config['min_liquidity_threshold']:
                return TokenCategory.GEM
            
            # High volatility = pump
            if abs(metrics.price_change_1h) > 0.3:
                return TokenCategory.PUMP
            
            # Low volatility = stable
            if abs(metrics.price_change_24h) < 0.1:
                return TokenCategory.STABLE
            
            # High volatility = volatile
            if abs(metrics.price_change_24h) > 0.5:
                return TokenCategory.VOLATILE
            
            return TokenCategory.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Error determining token category: {e}")
            return TokenCategory.UNKNOWN
    
    def _determine_opportunity_level(self, confidence_score: float, risk_score: float, pattern_signals: List[str]) -> OpportunityLevel:
        """Determine opportunity level"""
        try:
            # High confidence, low risk = high opportunity
            if confidence_score > 0.7 and risk_score < 0.3:
                return OpportunityLevel.HIGH
            
            # High risk = avoid
            if risk_score > 0.8:
                return OpportunityLevel.AVOID
            
            # Medium confidence, medium risk = medium opportunity
            if confidence_score > 0.5 and risk_score < 0.6:
                return OpportunityLevel.MEDIUM
            
            # Low confidence = low opportunity
            if confidence_score < 0.4:
                return OpportunityLevel.LOW
            
            # Default to monitor
            return OpportunityLevel.MONITOR
            
        except Exception as e:
            self.logger.error(f"Error determining opportunity level: {e}")
            return OpportunityLevel.MONITOR
    
    def _generate_recommendation(self, opportunity_level: OpportunityLevel, category: TokenCategory,
                               confidence_score: float, risk_score: float, pattern_signals: List[str],
                               risk_indicators: List[str]) -> Tuple[str, List[str]]:
        """Generate trading recommendation"""
        try:
            recommendation = "HOLD"
            reasoning = []
            
            if opportunity_level == OpportunityLevel.HIGH:
                recommendation = "BUY"
                reasoning.append("High confidence opportunity with low risk")
            elif opportunity_level == OpportunityLevel.MEDIUM:
                recommendation = "BUY_SMALL"
                reasoning.append("Moderate opportunity with acceptable risk")
            elif opportunity_level == OpportunityLevel.LOW:
                recommendation = "MONITOR"
                reasoning.append("Low confidence, wait for better signals")
            elif opportunity_level == OpportunityLevel.AVOID:
                recommendation = "AVOID"
                reasoning.append("High risk, avoid this token")
            else:  # MONITOR
                recommendation = "MONITOR"
                reasoning.append("Monitor for changes in conditions")
            
            # Add category-specific reasoning
            if category == TokenCategory.GEM:
                reasoning.append("Token shows gem-like characteristics")
            elif category == TokenCategory.RUG:
                reasoning.append("High rug pull risk detected")
            elif category == TokenCategory.PUMP:
                reasoning.append("Pump and dump pattern detected")
            
            # Add pattern-specific reasoning
            if "volume_spike" in pattern_signals:
                reasoning.append("Volume spike detected - potential momentum")
            if "price_momentum" in pattern_signals:
                reasoning.append("Price momentum building")
            if "low_liquidity" in risk_indicators:
                reasoning.append("Low liquidity - high slippage risk")
            
            return recommendation, reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            return "HOLD", ["Analysis error - defaulting to hold"]
    
    def _get_cached_analysis(self, token_address: str) -> Optional[TokenAnalysis]:
        """Get cached analysis if still valid"""
        try:
            if token_address in self.token_cache:
                analysis = self.token_cache[token_address]
                age_minutes = (datetime.now() - analysis.timestamp).total_seconds() / 60
                
                if age_minutes < self.config['cache_ttl_minutes']:
                    return analysis
                else:
                    # Remove expired cache
                    del self.token_cache[token_address]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached analysis: {e}")
            return None
    
    def _cache_analysis(self, token_address: str, analysis: TokenAnalysis):
        """Cache analysis result"""
        try:
            self.token_cache[token_address] = analysis
            
            # Limit cache size
            if len(self.token_cache) > 1000:
                # Remove oldest entries
                oldest_key = min(self.token_cache.keys(), 
                               key=lambda k: self.token_cache[k].timestamp)
                del self.token_cache[oldest_key]
                
        except Exception as e:
            self.logger.error(f"Error caching analysis: {e}")
    
    def _create_fallback_analysis(self, token_address: str) -> TokenAnalysis:
        """Create fallback analysis when main analysis fails"""
        try:
            metrics = TokenMetrics(
                token_address=token_address,
                symbol="UNKNOWN",
                price=0.0,
                market_cap=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                holder_count=0,
                age_hours=0.0,
                price_change_1h=0.0,
                price_change_24h=0.0,
                volume_change_24h=0.0,
                timestamp=datetime.now()
            )
            
            return TokenAnalysis(
                token_address=token_address,
                token_metrics=metrics,
                category=TokenCategory.UNKNOWN,
                opportunity_level=OpportunityLevel.MONITOR,
                confidence_score=0.0,
                risk_score=1.0,
                sentiment_score=0.5,
                technical_score=0.5,
                security_score=0.5,
                pattern_signals=[],
                risk_indicators=["analysis_failed"],
                recommendation="MONITOR",
                reasoning=["Analysis failed - insufficient data"],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating fallback analysis: {e}")
            raise
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while self.initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                current_time = datetime.now()
                expired_keys = []
                
                for token_address, analysis in self.token_cache.items():
                    age_minutes = (current_time - analysis.timestamp).total_seconds() / 60
                    if age_minutes > self.config['cache_ttl_minutes']:
                        expired_keys.append(token_address)
                
                # Remove expired entries
                for key in expired_keys:
                    del self.token_cache[key]
                
                if expired_keys:
                    self.logger.info(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _pattern_analysis_loop(self):
        """Background pattern analysis loop"""
        while self.initialized:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Analyze patterns in recent tokens
                await self._analyze_recent_patterns()
                
            except Exception as e:
                self.logger.error(f"Pattern analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _analyze_recent_patterns(self):
        """Analyze patterns in recently analyzed tokens"""
        try:
            # Get recent analyses
            recent_analyses = []
            for analyses in self.analysis_history.values():
                recent_analyses.extend(analyses[-5:])  # Last 5 analyses per token
            
            # Find common patterns
            pattern_counts = {}
            for analysis in recent_analyses:
                for pattern in analysis.pattern_signals:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Store pattern data
            self.pattern_database['recent_patterns'] = [
                {'pattern': pattern, 'count': count}
                for pattern, count in pattern_counts.items()
            ]
            
        except Exception as e:
            self.logger.error(f"Error analyzing recent patterns: {e}")
    
    def get_token_status(self, token_address: str) -> Dict[str, Any]:
        """Get token analysis status"""
        try:
            if token_address in self.analysis_history:
                recent_analyses = self.analysis_history[token_address]
                latest_analysis = recent_analyses[-1] if recent_analyses else None
                
                return {
                    'token_address': token_address,
                    'latest_analysis': {
                        'category': latest_analysis.category.value if latest_analysis else None,
                        'opportunity_level': latest_analysis.opportunity_level.value if latest_analysis else None,
                        'confidence_score': latest_analysis.confidence_score if latest_analysis else 0.0,
                        'risk_score': latest_analysis.risk_score if latest_analysis else 1.0,
                        'recommendation': latest_analysis.recommendation if latest_analysis else "UNKNOWN",
                        'timestamp': latest_analysis.timestamp.isoformat() if latest_analysis else None
                    },
                    'analysis_count': len(recent_analyses),
                    'cached': token_address in self.token_cache
                }
            else:
                return {
                    'token_address': token_address,
                    'latest_analysis': None,
                    'analysis_count': 0,
                    'cached': False
                }
                
        except Exception as e:
            self.logger.error(f"Error getting token status: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                'initialized': self.initialized,
                'cache_size': len(self.token_cache),
                'analyzed_tokens': len(self.analysis_history),
                'total_analyses': sum(len(analyses) for analyses in self.analysis_history.values()),
                'pattern_database_size': len(self.pattern_database)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the token intelligence system"""
        try:
            self.logger.info("🛑 Shutting down token intelligence system...")
            self.initialized = False
            self.logger.info("✅ Token intelligence system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global instance
_token_intelligence_system = None

async def get_token_intelligence_system() -> TokenIntelligenceSystem:
    """Get global token intelligence system instance"""
    global _token_intelligence_system
    if _token_intelligence_system is None:
        _token_intelligence_system = TokenIntelligenceSystem()
        await _token_intelligence_system.initialize()
    return _token_intelligence_system 