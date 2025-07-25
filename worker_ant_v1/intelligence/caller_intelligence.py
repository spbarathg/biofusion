"""
ADVANCED CALLER INTELLIGENCE SYSTEM
===================================

Real-time caller profiling and manipulation detection:
- Telegram/Discord/Twitter caller analysis
- Historical performance tracking
- Manipulation pattern detection
- Credibility scoring and risk assessment
Enhanced with TimescaleDB for high-performance time-series data storage.
"""

import asyncio
import time
import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import uuid

# TimescaleDB integration
from worker_ant_v1.core.database import (
    get_database_manager, 
    CallerProfile as DBCallerProfile
)
from worker_ant_v1.utils.logger import get_logger

class Platform(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    TWITTER = "twitter"
    REDDIT = "reddit"


class CredibilityLevel(Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class ManipulationRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CallAnalysis:
    """Individual call analysis result"""
    call_id: str
    caller_id: str
    token_address: str
    token_symbol: str
    call_timestamp: datetime
    platform: Platform
    call_text: str
    
    # Analysis results
    confidence_language: str = ""
    timeframe_mentioned: str = ""
    token_age_hours: float = 0.0
    liquidity_at_call: float = 0.0
    holder_count_at_call: int = 0
    
    # Outcome tracking
    outcome_tracked: bool = False
    max_profit_achieved: float = 0.0
    time_to_peak_minutes: int = 0
    final_outcome: str = ""  # profit, loss, rug, unknown
    pump_dump_likelihood: float = 0.0


@dataclass
class CallerProfile:
    """Enhanced caller profile with performance tracking"""
    caller_id: str
    username: str
    platform: Platform
    
    # Account information
    first_seen: datetime
    last_seen: datetime
    account_age_days: int
    follower_count: int
    verified_account: bool
    profile_completeness: float = 0.0
    
    # Performance metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    success_rate: float = 0.0
    avg_profit_percent: float = 0.0
    avg_time_to_profit_minutes: float = 0.0
    
    # Risk assessment
    credibility_level: CredibilityLevel = CredibilityLevel.UNKNOWN
    manipulation_risk: ManipulationRisk = ManipulationRisk.LOW
    trust_score: float = 0.0
    risk_indicators: List[str] = None
    
    # Social metrics
    engagement_rate: float = 0.0
    sentiment_alignment: float = 0.0
    network_connections: List[str] = None
    associated_groups: List[str] = None
    
    # Historical data
    recent_calls: List[Dict[str, Any]] = None
    performance_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.risk_indicators is None:
            self.risk_indicators = []
        if self.network_connections is None:
            self.network_connections = []
        if self.associated_groups is None:
            self.associated_groups = []
        if self.recent_calls is None:
            self.recent_calls = []
        if self.performance_history is None:
            self.performance_history = []

    def to_db_profile(self) -> DBCallerProfile:
        """Convert to TimescaleDB caller profile"""
        return DBCallerProfile(
            timestamp=datetime.utcnow(),
            caller_id=self.caller_id,
            username=self.username,
            platform=self.platform.value,
            account_age_days=self.account_age_days,
            follower_count=self.follower_count,
            verified_account=self.verified_account,
            total_calls=self.total_calls,
            successful_calls=self.successful_calls,
            success_rate=self.success_rate,
            avg_profit_percent=self.avg_profit_percent,
            trust_score=self.trust_score,
            credibility_level=self.credibility_level.value,
            manipulation_risk=self.manipulation_risk.value,
            risk_indicators=self.risk_indicators,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            profile_data={
                'profile_completeness': self.profile_completeness,
                'failed_calls': self.failed_calls,
                'avg_time_to_profit_minutes': self.avg_time_to_profit_minutes,
                'engagement_rate': self.engagement_rate,
                'sentiment_alignment': self.sentiment_alignment,
                'network_connections': self.network_connections,
                'associated_groups': self.associated_groups,
                'recent_calls': self.recent_calls[-10:],  # Keep last 10
                'performance_history': self.performance_history[-50:]  # Keep last 50
            }
        )


class AdvancedCallerIntelligence:
    """Advanced caller intelligence with TimescaleDB integration"""
    
    def __init__(self):
        self.logger = get_logger("AdvancedCallerIntelligence")
        
        # TimescaleDB manager
        self.db_manager = None
        
        # In-memory cache for active profiles
        self.caller_profiles: Dict[str, CallerProfile] = {}
        self.call_analysis_cache: Dict[str, CallAnalysis] = {}
        
        # Configuration
        self.config = {
            'max_cache_size': 1000,
            'cache_expiry_hours': 24,
            'min_credibility_threshold': 0.3,
            'manipulation_detection_enabled': True,
            'real_time_tracking': True
        }
        
        # Analysis weights and thresholds
        self.credibility_weights = {
            'account_age': 0.2,
            'follower_count': 0.15,
            'verified_status': 0.25,
            'success_rate': 0.3,
            'engagement_rate': 0.1
        }
        
        # Tracking state
        self.initialized = False
        self.monitoring_active = False

    async def initialize(self) -> bool:
        """Initialize the caller intelligence system"""
        try:
            self.logger.info("🧠 Initializing Advanced Caller Intelligence...")
            
            # Initialize TimescaleDB manager
            self.db_manager = await get_database_manager()
            
            # Load existing profiles from cache
            await self._load_cached_profiles()
            
            self.initialized = True
            self.monitoring_active = True
            
            self.logger.info("✅ Advanced Caller Intelligence initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize caller intelligence: {e}")
            return False

    async def _load_cached_profiles(self):
        """Load frequently accessed profiles into memory cache"""
        try:
            # Get recent caller profiles from TimescaleDB
            # We'll load profiles that have been active in the last 24 hours
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            
            # For now, we'll start with empty cache and load profiles as needed
            # In a production system, you might want to preload top performers
            self.logger.info("📊 Caller profiles cache initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading cached profiles: {e}")

    async def analyze_caller_signal(self, 
                                   caller_id: str,
                                   username: str,
                                   platform: Platform,
                                   call_text: str,
                                   token_address: str,
                                   token_symbol: str,
                                   additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze a caller signal and update intelligence"""
        try:
            call_id = str(uuid.uuid4())
            call_timestamp = datetime.utcnow()
            
            # Get or create caller profile
            profile = await self._get_or_create_caller_profile(
                caller_id, username, platform, additional_data or {}
            )
            
            # Perform call analysis
            analysis = await self._analyze_call_content(
                call_id, caller_id, token_address, token_symbol,
                call_text, platform, call_timestamp
            )
            
            # Update caller profile with new call
            await self._update_caller_with_call(profile, analysis)
            
            # Save updated profile to TimescaleDB
            await self._save_caller_profile(profile)
            
            # Cache the analysis for outcome tracking
            self.call_analysis_cache[call_id] = analysis
            
            # Generate intelligence report
            intelligence_report = {
                'call_id': call_id,
                'caller_id': caller_id,
                'credibility_score': profile.trust_score,
                'credibility_level': profile.credibility_level.value,
                'manipulation_risk': profile.manipulation_risk.value,
                'historical_success_rate': profile.success_rate,
                'avg_profit_potential': profile.avg_profit_percent,
                'risk_indicators': profile.risk_indicators,
                'recommended_action': self._generate_recommendation(profile, analysis),
                'confidence_score': self._calculate_confidence_score(profile, analysis),
                'call_analysis': {
                    'confidence_language': analysis.confidence_language,
                    'timeframe_mentioned': analysis.timeframe_mentioned,
                    'pump_dump_likelihood': analysis.pump_dump_likelihood
                }
            }
            
            self.logger.info(f"📊 Analyzed signal from {username}: {intelligence_report['recommended_action']}")
            
            return intelligence_report
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing caller signal: {e}")
            return {'error': str(e)}

    async def _get_or_create_caller_profile(self, 
                                          caller_id: str,
                                          username: str,
                                          platform: Platform,
                                          additional_data: Dict[str, Any]) -> CallerProfile:
        """Get existing caller profile or create new one"""
        
        # Check cache first
        if caller_id in self.caller_profiles:
            profile = self.caller_profiles[caller_id]
            profile.last_seen = datetime.utcnow()
            return profile
        
        # Try to load from TimescaleDB
        try:
            # Since TimescaleDB stores time-series data, we need to get the latest profile
            # This would typically involve a query for the most recent record
            # For now, we'll create a new profile and let the system learn
            
            profile = CallerProfile(
                caller_id=caller_id,
                username=username,
                platform=platform,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                account_age_days=additional_data.get('account_age_days', 0),
                follower_count=additional_data.get('follower_count', 0),
                verified_account=additional_data.get('verified_account', False)
            )
            
            # Calculate initial metrics
            await self._calculate_initial_credibility(profile, additional_data)
            
            # Cache the profile
            self.caller_profiles[caller_id] = profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Error loading caller profile: {e}")
            # Return default profile
            return CallerProfile(
                caller_id=caller_id,
                username=username,
                platform=platform,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                account_age_days=0,
                follower_count=0,
                verified_account=False
            )

    async def _analyze_call_content(self,
                                   call_id: str,
                                   caller_id: str,
                                   token_address: str,
                                   token_symbol: str,
                                   call_text: str,
                                   platform: Platform,
                                   call_timestamp: datetime) -> CallAnalysis:
        """Analyze the content of a call for manipulation patterns"""
        
        analysis = CallAnalysis(
            call_id=call_id,
            caller_id=caller_id,
            token_address=token_address,
            token_symbol=token_symbol,
            call_timestamp=call_timestamp,
            platform=platform,
            call_text=call_text
        )
        
        # Analyze confidence language
        analysis.confidence_language = self._detect_confidence_language(call_text)
        
        # Extract timeframe mentions
        analysis.timeframe_mentioned = self._extract_timeframe(call_text)
        
        # Calculate pump and dump likelihood
        analysis.pump_dump_likelihood = self._calculate_pump_dump_likelihood(call_text)
        
        return analysis

    def _detect_confidence_language(self, text: str) -> str:
        """Detect confidence level in language"""
        text_lower = text.lower()
        
        high_confidence_patterns = [
            r'guaranteed?', r'100%', r'for sure', r'definitely',
            r'moon', r'pump', r'rocket', r'lambo'
        ]
        
        medium_confidence_patterns = [
            r'likely', r'probably', r'should', r'expecting',
            r'potential', r'could', r'might'
        ]
        
        low_confidence_patterns = [
            r'maybe', r'possibly', r'uncertain', r'risky',
            r'dyor', r'not financial advice'
        ]
        
        high_matches = sum(1 for pattern in high_confidence_patterns 
                          if re.search(pattern, text_lower))
        medium_matches = sum(1 for pattern in medium_confidence_patterns 
                            if re.search(pattern, text_lower))
        low_matches = sum(1 for pattern in low_confidence_patterns 
                         if re.search(pattern, text_lower))
        
        if high_matches > medium_matches and high_matches > low_matches:
            return "high"
        elif medium_matches > low_matches:
            return "medium"
        else:
            return "low"

    def _extract_timeframe(self, text: str) -> str:
        """Extract mentioned timeframes from call text"""
        timeframe_patterns = {
            r'(\d+)\s*(minute|min)s?': 'minutes',
            r'(\d+)\s*(hour|hr)s?': 'hours',
            r'(\d+)\s*(day)s?': 'days',
            r'(\d+)\s*(week)s?': 'weeks',
            r'immediately|now|right now': 'immediate',
            r'soon|shortly': 'short_term',
            r'long.?term|hodl': 'long_term'
        }
        
        text_lower = text.lower()
        for pattern, timeframe in timeframe_patterns.items():
            if re.search(pattern, text_lower):
                return timeframe
        
        return "unspecified"

    def _calculate_pump_dump_likelihood(self, text: str) -> float:
        """Calculate likelihood of pump and dump scheme"""
        pump_dump_indicators = [
            r'pump', r'moon', r'rocket', r'lambo',
            r'100x', r'1000x', r'guaranteed',
            r'last chance', r'urgent', r'now or never',
            r'secret', r'insider', r'exclusive'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in pump_dump_indicators 
                     if re.search(pattern, text_lower))
        
        # Normalize to 0-1 scale
        likelihood = min(matches / len(pump_dump_indicators), 1.0)
        
        return likelihood

    async def _calculate_initial_credibility(self, 
                                           profile: CallerProfile,
                                           additional_data: Dict[str, Any]):
        """Calculate initial credibility score for new profiles"""
        score = 0.0
        
        # Account age factor
        if profile.account_age_days > 365:
            score += self.credibility_weights['account_age'] * 1.0
        elif profile.account_age_days > 180:
            score += self.credibility_weights['account_age'] * 0.7
        elif profile.account_age_days > 30:
            score += self.credibility_weights['account_age'] * 0.4
        
        # Follower count factor  
        if profile.follower_count > 10000:
            score += self.credibility_weights['follower_count'] * 1.0
        elif profile.follower_count > 1000:
            score += self.credibility_weights['follower_count'] * 0.7
        elif profile.follower_count > 100:
            score += self.credibility_weights['follower_count'] * 0.4
        
        # Verified status
        if profile.verified_account:
            score += self.credibility_weights['verified_status']
        
        profile.trust_score = score
        
        # Set credibility level based on score
        if score >= 0.8:
            profile.credibility_level = CredibilityLevel.HIGH
        elif score >= 0.5:
            profile.credibility_level = CredibilityLevel.MEDIUM
        elif score >= 0.2:
            profile.credibility_level = CredibilityLevel.LOW
        else:
            profile.credibility_level = CredibilityLevel.UNKNOWN

    async def _update_caller_with_call(self, profile: CallerProfile, analysis: CallAnalysis):
        """Update caller profile with new call information"""
        profile.total_calls += 1
        profile.last_seen = datetime.utcnow()
        
        # Add to recent calls
        call_summary = {
            'call_id': analysis.call_id,
            'timestamp': analysis.call_timestamp.isoformat(),
            'token_symbol': analysis.token_symbol,
            'confidence_language': analysis.confidence_language,
            'pump_dump_likelihood': analysis.pump_dump_likelihood
        }
        
        profile.recent_calls.append(call_summary)
        
        # Keep only last 20 calls
        if len(profile.recent_calls) > 20:
            profile.recent_calls = profile.recent_calls[-20:]
        
        # Update manipulation risk based on pump/dump likelihood
        if analysis.pump_dump_likelihood > 0.7:
            profile.manipulation_risk = ManipulationRisk.HIGH
            if 'high_pump_dump_language' not in profile.risk_indicators:
                profile.risk_indicators.append('high_pump_dump_language')
        elif analysis.pump_dump_likelihood > 0.4:
            profile.manipulation_risk = ManipulationRisk.MEDIUM

    async def _save_caller_profile(self, profile: CallerProfile):
        """Save caller profile to TimescaleDB"""
        try:
            db_profile = profile.to_db_profile()
            await self.db_manager.insert_caller_profile(db_profile)
            
        except Exception as e:
            self.logger.error(f"❌ Error saving caller profile: {e}")

    def _generate_recommendation(self, profile: CallerProfile, analysis: CallAnalysis) -> str:
        """Generate trading recommendation based on analysis"""
        
        if profile.manipulation_risk == ManipulationRisk.HIGH:
            return "AVOID - High manipulation risk detected"
        
        if profile.credibility_level == CredibilityLevel.HIGH and profile.success_rate > 0.7:
            return "STRONG_BUY - High credibility caller with good track record"
        
        if profile.credibility_level == CredibilityLevel.MEDIUM and profile.success_rate > 0.5:
            return "MODERATE_BUY - Medium credibility, proceed with caution"
        
        if profile.credibility_level == CredibilityLevel.LOW or profile.success_rate < 0.3:
            return "WEAK_SIGNAL - Low credibility or poor performance"
        
        if analysis.pump_dump_likelihood > 0.6:
            return "PUMP_DUMP_RISK - High likelihood of manipulation"
        
        return "NEUTRAL - Insufficient data for strong recommendation"

    def _calculate_confidence_score(self, profile: CallerProfile, analysis: CallAnalysis) -> float:
        """Calculate confidence score for the recommendation"""
        base_score = profile.trust_score
        
        # Adjust based on call history
        if profile.total_calls > 10:
            base_score += 0.1
        if profile.total_calls > 50:
            base_score += 0.1
        
        # Reduce confidence for high manipulation risk
        if analysis.pump_dump_likelihood > 0.5:
            base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))

    async def track_call_outcome(self, 
                                call_id: str,
                                outcome: str,
                                profit_percent: float = 0.0,
                                time_to_peak_minutes: int = 0) -> bool:
        """Track the outcome of a previously analyzed call"""
        try:
            if call_id not in self.call_analysis_cache:
                self.logger.warning(f"⚠️ Call ID {call_id} not found in cache")
                return False
            
            analysis = self.call_analysis_cache[call_id]
            caller_id = analysis.caller_id
            
            # Update analysis with outcome
            analysis.outcome_tracked = True
            analysis.final_outcome = outcome
            analysis.max_profit_achieved = profit_percent
            analysis.time_to_peak_minutes = time_to_peak_minutes
            
            # Update caller profile
            if caller_id in self.caller_profiles:
                profile = self.caller_profiles[caller_id]
                await self._update_caller_with_outcome(profile, analysis)
                await self._save_caller_profile(profile)
            
            self.logger.info(f"📊 Tracked outcome for call {call_id}: {outcome} ({profit_percent:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error tracking call outcome: {e}")
            return False

    async def _update_caller_with_outcome(self, profile: CallerProfile, call_analysis: CallAnalysis):
        """Update caller profile with call outcome"""
        
        profile.total_calls += 1
        profile.last_seen = datetime.utcnow()
        
        # Determine if call was successful
        if call_analysis.final_outcome == 'profit' and call_analysis.max_profit_achieved > 5:
            profile.successful_calls += 1
            
            # Update performance metrics
            if profile.avg_profit_percent == 0:
                profile.avg_profit_percent = call_analysis.max_profit_achieved
            else:
                profile.avg_profit_percent = (
                    profile.avg_profit_percent * 0.8 + 
                    call_analysis.max_profit_achieved * 0.2
                )
        else:
            profile.failed_calls += 1
        
        # Update success rate
        if profile.total_calls > 0:
            profile.success_rate = profile.successful_calls / profile.total_calls
        
        # Add to performance history
        profile.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'call_id': call_analysis.call_id,
            'outcome': call_analysis.final_outcome,
            'profit': call_analysis.max_profit_achieved
        })
        
        # Keep only recent history
        if len(profile.performance_history) > 50:
            profile.performance_history = profile.performance_history[-50:]
        
        # Recalculate trust score based on updated performance
        await self._recalculate_trust_score(profile)

    async def _recalculate_trust_score(self, profile: CallerProfile):
        """Recalculate trust score based on updated performance"""
        score = 0.0
        
        # Base credibility factors (same as initial calculation)
        if profile.account_age_days > 365:
            score += self.credibility_weights['account_age'] * 1.0
        elif profile.account_age_days > 180:
            score += self.credibility_weights['account_age'] * 0.7
        elif profile.account_age_days > 30:
            score += self.credibility_weights['account_age'] * 0.4
        
        if profile.follower_count > 10000:
            score += self.credibility_weights['follower_count'] * 1.0
        elif profile.follower_count > 1000:
            score += self.credibility_weights['follower_count'] * 0.7
        elif profile.follower_count > 100:
            score += self.credibility_weights['follower_count'] * 0.4
        
        if profile.verified_account:
            score += self.credibility_weights['verified_status']
        
        # Performance-based factors
        if profile.total_calls >= 5:  # Need minimum calls for performance evaluation
            score += self.credibility_weights['success_rate'] * profile.success_rate
        
        # Engagement rate factor
        score += self.credibility_weights['engagement_rate'] * profile.engagement_rate
        
        profile.trust_score = min(1.0, score)
        
        # Update credibility level
        if profile.trust_score >= 0.8:
            profile.credibility_level = CredibilityLevel.HIGH
        elif profile.trust_score >= 0.5:
            profile.credibility_level = CredibilityLevel.MEDIUM
        elif profile.trust_score >= 0.2:
            profile.credibility_level = CredibilityLevel.LOW
        else:
            profile.credibility_level = CredibilityLevel.UNKNOWN

    async def get_caller_intelligence(self, caller_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive intelligence report for a caller"""
        try:
            if caller_id in self.caller_profiles:
                profile = self.caller_profiles[caller_id]
                
                return {
                    'caller_id': caller_id,
                    'username': profile.username,
                    'platform': profile.platform.value,
                    'credibility_level': profile.credibility_level.value,
                    'trust_score': profile.trust_score,
                    'success_rate': profile.success_rate,
                    'total_calls': profile.total_calls,
                    'avg_profit_percent': profile.avg_profit_percent,
                    'manipulation_risk': profile.manipulation_risk.value,
                    'risk_indicators': profile.risk_indicators,
                    'account_age_days': profile.account_age_days,
                    'follower_count': profile.follower_count,
                    'verified_account': profile.verified_account,
                    'recent_performance': profile.performance_history[-10:],
                    'last_seen': profile.last_seen.isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting caller intelligence: {e}")
            return None

    async def get_top_performers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing callers"""
        try:
            # Sort cached profiles by performance
            sorted_profiles = sorted(
                self.caller_profiles.values(),
                key=lambda p: (p.success_rate * p.trust_score, p.total_calls),
                reverse=True
            )
            
            top_performers = []
            for profile in sorted_profiles[:limit]:
                if profile.total_calls >= 5:  # Minimum call threshold
                    top_performers.append({
                        'caller_id': profile.caller_id,
                        'username': profile.username,
                        'platform': profile.platform.value,
                        'success_rate': profile.success_rate,
                        'trust_score': profile.trust_score,
                        'total_calls': profile.total_calls,
                        'avg_profit_percent': profile.avg_profit_percent
                    })
            
            return top_performers
            
        except Exception as e:
            self.logger.error(f"❌ Error getting top performers: {e}")
            return []

    async def shutdown(self):
        """Shutdown the caller intelligence system"""
        try:
            self.logger.info("🛑 Shutting down Advanced Caller Intelligence...")
            
            # Save any remaining cached profiles
            for profile in self.caller_profiles.values():
                await self._save_caller_profile(profile)
            
            self.monitoring_active = False
            self.logger.info("✅ Advanced Caller Intelligence shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
