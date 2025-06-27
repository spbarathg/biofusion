"""
ADVANCED CALLER INTELLIGENCE SYSTEM
===================================

Real-time caller profiling and manipulation detection:
- Telegram/Discord/Twitter caller analysis
- Historical performance tracking
- Manipulation pattern detection
- Credibility scoring and risk assessment
"""

import asyncio
import time
import json
# Standard library imports already available
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict

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
    ELITE = "elite"
    BLACKLISTED = "blacklisted"

class ManipulationRisk(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class CallerProfile:
    """Comprehensive caller profile"""
    caller_id: str
    username: str
    platform: Platform
    first_seen: datetime
    last_seen: datetime
    
    # Verification metrics
    account_age_days: int
    follower_count: int
    verified_account: bool
    profile_completeness: float
    
    # Performance metrics
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float
    avg_profit_percent: float
    avg_time_to_profit_minutes: float
    
    # Risk metrics
    credibility_level: CredibilityLevel
    manipulation_risk: ManipulationRisk
    trust_score: float
    risk_indicators: List[str]
    
    # Social metrics
    engagement_rate: float
    sentiment_alignment: float
    network_connections: List[str]
    associated_groups: List[str]
    
    # Call history
    recent_calls: List[Dict[str, Any]]
    performance_history: List[Dict[str, Any]]

@dataclass
class CallAnalysis:
    """Analysis of specific call/recommendation"""
    call_id: str
    caller_id: str
    token_address: str
    token_symbol: str
    call_timestamp: datetime
    platform: Platform
    
    # Call content analysis
    call_text: str
    confidence_language: float
    urgency_indicators: List[str]
    price_predictions: List[float]
    timeframe_mentioned: Optional[str]
    
    # Context analysis
    market_conditions: Dict[str, Any]
    token_age_hours: float
    liquidity_at_call: float
    holder_count_at_call: int
    
    # Outcome tracking
    outcome_tracked: bool
    max_profit_achieved: Optional[float]
    time_to_peak_minutes: Optional[int]
    final_outcome: Optional[str]
    
    # Risk assessment
    manipulation_indicators: List[str]
    coordination_signals: List[str]
    pump_dump_likelihood: float

class AdvancedCallerIntelligence:
    """Advanced caller intelligence and manipulation detection"""
    
    def __init__(self, database_path: str = "caller_intelligence.db"):
        self.database_path = database_path
        self.caller_profiles = {}
        self.call_analysis_cache = deque(maxlen=10000)
        
        # Analysis parameters
        self.min_calls_for_rating = 5
        self.credibility_threshold = 0.7
        self.manipulation_threshold = 0.6
        
        # Pattern detection keywords
        self.pump_keywords = [
            'moon', 'rocket', 'gem', 'hidden', 'low cap', 'early',
            'guaranteed', 'sure thing', 'can\'t miss', 'insider',
            '100x', '1000x', 'next shib', 'next doge'
        ]
        
        self.urgency_keywords = [
            'now', 'hurry', 'quick', 'fast', 'limited', 'urgent',
            'don\'t miss', 'last chance', 'act now', 'fomo'
        ]
        
        self.coordination_patterns = [
            r'everyone buy at \d+:\d+',
            r'coordinate(d)? buy',
            r'pump together',
            r'all in at once'
        ]

    async def initialize(self):
        """Initialize caller intelligence system"""
        await self._setup_database()
        await self._load_existing_profiles()
        
    async def _setup_database(self):
        """Setup SQLite database for intelligence storage"""
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Caller profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS caller_profiles (
                caller_id TEXT PRIMARY KEY,
                username TEXT,
                platform TEXT,
                first_seen TEXT,
                last_seen TEXT,
                account_age_days INTEGER,
                follower_count INTEGER,
                verified_account BOOLEAN,
                total_calls INTEGER,
                successful_calls INTEGER,
                success_rate REAL,
                credibility_level TEXT,
                manipulation_risk TEXT,
                trust_score REAL,
                profile_data TEXT
            )
        ''')
        
        # Call analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_analysis (
                call_id TEXT PRIMARY KEY,
                caller_id TEXT,
                token_address TEXT,
                token_symbol TEXT,
                call_timestamp TEXT,
                platform TEXT,
                call_text TEXT,
                outcome_tracked BOOLEAN,
                max_profit_achieved REAL,
                final_outcome TEXT,
                analysis_data TEXT,
                FOREIGN KEY (caller_id) REFERENCES caller_profiles (caller_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def analyze_caller(self, caller_data: Dict[str, Any]) -> CallerProfile:
        """Analyze caller and return comprehensive profile"""
        
        caller_id = caller_data.get('id') or caller_data.get('username', 'unknown')
        
        # Check if profile exists
        if caller_id in self.caller_profiles:
            profile = self.caller_profiles[caller_id]
            await self._update_caller_profile(profile, caller_data)
        else:
            profile = await self._create_caller_profile(caller_data)
            self.caller_profiles[caller_id] = profile
        
        # Analyze recent activity
        await self._analyze_recent_activity(profile)
        
        # Update credibility and risk scores
        await self._update_credibility_scores(profile)
        
        # Save to database
        await self._save_caller_profile(profile)
        
        return profile
    
    async def _create_caller_profile(self, caller_data: Dict[str, Any]) -> CallerProfile:
        """Create new caller profile"""
        
        caller_id = caller_data.get('id') or caller_data.get('username', 'unknown')
        username = caller_data.get('username', 'anonymous')
        platform = Platform(caller_data.get('platform', 'telegram'))
        
        # Extract account metrics
        account_age = caller_data.get('account_age_days', 0)
        followers = caller_data.get('followers', 0)
        verified = caller_data.get('verified', False)
        
        # Calculate profile completeness
        completeness = self._calculate_profile_completeness(caller_data)
        
        return CallerProfile(
            caller_id=caller_id,
            username=username,
            platform=platform,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            account_age_days=account_age,
            follower_count=followers,
            verified_account=verified,
            profile_completeness=completeness,
            total_calls=0,
            successful_calls=0,
            failed_calls=0,
            success_rate=0.0,
            avg_profit_percent=0.0,
            avg_time_to_profit_minutes=0.0,
            credibility_level=CredibilityLevel.UNKNOWN,
            manipulation_risk=ManipulationRisk.MEDIUM,
            trust_score=0.5,
            risk_indicators=[],
            engagement_rate=0.0,
            sentiment_alignment=0.0,
            network_connections=[],
            associated_groups=[],
            recent_calls=[],
            performance_history=[]
        )
    
    async def analyze_call(self, call_data: Dict[str, Any]) -> CallAnalysis:
        """Analyze specific call/recommendation"""
        
        call_id = call_data.get('id') or f"call_{int(time.time())}"
        caller_id = call_data.get('caller_id', 'unknown')
        
        # Extract call information
        token_address = call_data.get('token_address', '')
        token_symbol = call_data.get('token_symbol', '')
        call_text = call_data.get('text', '')
        platform = Platform(call_data.get('platform', 'telegram'))
        
        # Analyze call content
        confidence_language = await self._analyze_confidence_language(call_text)
        urgency_indicators = await self._detect_urgency_indicators(call_text)
        price_predictions = await self._extract_price_predictions(call_text)
        
        # Detect manipulation indicators
        manipulation_indicators = await self._detect_manipulation_indicators(call_text)
        coordination_signals = await self._detect_coordination_signals(call_text)
        pump_dump_likelihood = await self._calculate_pump_dump_likelihood(call_text, call_data)
        
        # Create call analysis
        analysis = CallAnalysis(
            call_id=call_id,
            caller_id=caller_id,
            token_address=token_address,
            token_symbol=token_symbol,
            call_timestamp=datetime.now(),
            platform=platform,
            call_text=call_text,
            confidence_language=confidence_language,
            urgency_indicators=urgency_indicators,
            price_predictions=price_predictions,
            timeframe_mentioned=await self._extract_timeframe(call_text),
            market_conditions=await self._get_market_conditions(),
            token_age_hours=call_data.get('token_age_hours', 0),
            liquidity_at_call=call_data.get('liquidity', 0),
            holder_count_at_call=call_data.get('holder_count', 0),
            outcome_tracked=False,
            max_profit_achieved=None,
            time_to_peak_minutes=None,
            final_outcome=None,
            manipulation_indicators=manipulation_indicators,
            coordination_signals=coordination_signals,
            pump_dump_likelihood=pump_dump_likelihood
        )
        
        # Cache analysis
        self.call_analysis_cache.append(analysis)
        
        # Save to database
        await self._save_call_analysis(analysis)
        
        return analysis
    
    async def _analyze_confidence_language(self, text: str) -> float:
        """Analyze confidence level in language"""
        
        text_lower = text.lower()
        
        # High confidence indicators
        high_confidence = ['guarantee', 'sure', 'certain', 'definite', '100%', 'no doubt']
        high_count = sum(1 for phrase in high_confidence if phrase in text_lower)
        
        # Medium confidence indicators
        medium_confidence = ['likely', 'probably', 'expect', 'should', 'think']
        medium_count = sum(1 for phrase in medium_confidence if phrase in text_lower)
        
        # Low confidence indicators
        low_confidence = ['maybe', 'might', 'could', 'possibly', 'perhaps']
        low_count = sum(1 for phrase in low_confidence if phrase in text_lower)
        
        # Calculate confidence score
        total_indicators = high_count + medium_count + low_count
        if total_indicators == 0:
            return 0.5  # Neutral
        
        confidence_score = (high_count * 1.0 + medium_count * 0.6 + low_count * 0.2) / total_indicators
        return min(confidence_score, 1.0)
    
    async def _detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators in text"""
        
        text_lower = text.lower()
        found_indicators = []
        
        for keyword in self.urgency_keywords:
            if keyword in text_lower:
                found_indicators.append(keyword)
        
        # Check for time-based urgency
        time_patterns = [
            r'\d+ minutes?',
            r'\d+ hours?',
            r'ending soon',
            r'limited time'
        ]
        
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                found_indicators.append(f"time_urgency: {pattern}")
        
        return found_indicators
    
    async def _extract_price_predictions(self, text: str) -> List[float]:
        """Extract price predictions from text"""
        
        predictions = []
        
        # Look for percentage predictions
        percent_pattern = r'(\d+)x|\+(\d+)%|(\d+)%\s*gain'
        matches = re.findall(percent_pattern, text, re.IGNORECASE)
        
        for match in matches:
            for group in match:
                if group:
                    predictions.append(float(group))
        
        # Look for price targets
        price_pattern = r'\$(\d+\.?\d*)'
        price_matches = re.findall(price_pattern, text)
        
        for price in price_matches:
            predictions.append(float(price))
        
        return predictions
    
    async def _detect_manipulation_indicators(self, text: str) -> List[str]:
        """Detect manipulation indicators in call text"""
        
        text_lower = text.lower()
        indicators = []
        
        # Check for pump keywords
        for keyword in self.pump_keywords:
            if keyword in text_lower:
                indicators.append(f"pump_keyword: {keyword}")
        
        # Check for financial advice disclaimers
        if 'not financial advice' not in text_lower and 'nfa' not in text_lower:
            indicators.append("no_disclaimer")
        
        # Check for excessive certainty
        certainty_keywords = ['guarantee', 'sure thing', 'can\'t lose', 'risk-free']
        for keyword in certainty_keywords:
            if keyword in text_lower:
                indicators.append(f"excessive_certainty: {keyword}")
        
        # Check for FOMO tactics
        fomo_patterns = [
            r'last chance',
            r'don\'t miss out',
            r'everyone is buying',
            r'going viral'
        ]
        
        for pattern in fomo_patterns:
            if re.search(pattern, text_lower):
                indicators.append(f"fomo_tactic: {pattern}")
        
        return indicators
    
    async def _detect_coordination_signals(self, text: str) -> List[str]:
        """Detect coordination signals"""
        
        text_lower = text.lower()
        signals = []
        
        # Check for coordination patterns
        for pattern in self.coordination_patterns:
            if re.search(pattern, text_lower):
                signals.append(f"coordination_pattern: {pattern}")
        
        return signals
    
    async def _calculate_pump_dump_likelihood(self, text: str, call_data: Dict[str, Any]) -> float:
        """Calculate likelihood this is a pump and dump"""
        
        likelihood = 0.0
        
        # Text analysis factors
        manipulation_count = len(await self._detect_manipulation_indicators(text))
        urgency_count = len(await self._detect_urgency_indicators(text))
        coordination_count = len(await self._detect_coordination_signals(text))
        
        # Token factors
        token_age = call_data.get('token_age_hours', 24)
        liquidity = call_data.get('liquidity', 1000)
        holder_count = call_data.get('holder_count', 100)
        
        # Calculate base likelihood from text
        text_likelihood = min((manipulation_count * 0.2 + urgency_count * 0.15 + coordination_count * 0.3), 0.8)
        
        # Adjust for token characteristics
        if token_age < 24:  # Very new token
            text_likelihood += 0.2
        
        if liquidity < 1000:  # Low liquidity
            text_likelihood += 0.15
        
        if holder_count < 50:  # Few holders
            text_likelihood += 0.1
        
        return min(text_likelihood, 1.0)
    
    async def track_call_outcome(self, call_id: str, outcome_data: Dict[str, Any]):
        """Track the outcome of a specific call"""
        
        # Find the call analysis
        call_analysis = None
        for analysis in self.call_analysis_cache:
            if analysis.call_id == call_id:
                call_analysis = analysis
                break
        
        if not call_analysis:
            return
        
        # Update outcome data
        call_analysis.outcome_tracked = True
        call_analysis.max_profit_achieved = outcome_data.get('max_profit_percent', 0)
        call_analysis.time_to_peak_minutes = outcome_data.get('time_to_peak_minutes', 0)
        call_analysis.final_outcome = outcome_data.get('final_outcome', 'unknown')
        
        # Update caller profile
        if call_analysis.caller_id in self.caller_profiles:
            profile = self.caller_profiles[call_analysis.caller_id]
            await self._update_caller_with_outcome(profile, call_analysis)
        
        # Save updated analysis
        await self._save_call_analysis(call_analysis)
    
    async def _update_caller_with_outcome(self, profile: CallerProfile, call_analysis: CallAnalysis):
        """Update caller profile with call outcome"""
        
        profile.total_calls += 1
        profile.last_seen = datetime.now()
        
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
            'timestamp': datetime.now().isoformat(),
            'call_id': call_analysis.call_id,
            'outcome': call_analysis.final_outcome,
            'profit': call_analysis.max_profit_achieved
        })
        
        # Keep only recent history
        if len(profile.performance_history) > 50:
            profile.performance_history = profile.performance_history[-50:]
    
    async def _update_credibility_scores(self, profile: CallerProfile):
        """Update credibility and risk scores"""
        
        # Base credibility from success rate
        base_credibility = profile.success_rate if profile.total_calls >= self.min_calls_for_rating else 0.5
        
        # Adjust for account legitimacy
        legitimacy_boost = 0
        if profile.verified_account:
            legitimacy_boost += 0.1
        if profile.account_age_days > 365:
            legitimacy_boost += 0.1
        if profile.follower_count > 1000:
            legitimacy_boost += 0.05
        
        # Calculate trust score
        profile.trust_score = min(base_credibility + legitimacy_boost, 1.0)
        
        # Determine credibility level
        if profile.trust_score >= 0.8:
            profile.credibility_level = CredibilityLevel.ELITE
        elif profile.trust_score >= 0.7:
            profile.credibility_level = CredibilityLevel.HIGH
        elif profile.trust_score >= 0.5:
            profile.credibility_level = CredibilityLevel.MEDIUM
        elif profile.trust_score >= 0.3:
            profile.credibility_level = CredibilityLevel.LOW
        else:
            profile.credibility_level = CredibilityLevel.UNKNOWN
        
        # Assess manipulation risk
        risk_score = await self._calculate_manipulation_risk(profile)
        
        if risk_score >= 0.8:
            profile.manipulation_risk = ManipulationRisk.EXTREME
        elif risk_score >= 0.6:
            profile.manipulation_risk = ManipulationRisk.HIGH
        elif risk_score >= 0.4:
            profile.manipulation_risk = ManipulationRisk.MEDIUM
        elif risk_score >= 0.2:
            profile.manipulation_risk = ManipulationRisk.LOW
        else:
            profile.manipulation_risk = ManipulationRisk.MINIMAL
    
    async def _calculate_manipulation_risk(self, profile: CallerProfile) -> float:
        """Calculate manipulation risk score"""
        
        risk_score = 0.0
        
        # Check recent calls for manipulation indicators
        recent_calls_with_indicators = 0
        total_recent_calls = len(profile.recent_calls)
        
        for call_data in profile.recent_calls[-10:]:
            if call_data.get('manipulation_indicators', []):
                recent_calls_with_indicators += 1
        
        if total_recent_calls > 0:
            risk_score += (recent_calls_with_indicators / total_recent_calls) * 0.4
        
        # Account age factor
        if profile.account_age_days < 30:
            risk_score += 0.3
        elif profile.account_age_days < 90:
            risk_score += 0.2
        
        # Success rate factor (too high can be suspicious)
        if profile.success_rate > 0.9 and profile.total_calls > 5:
            risk_score += 0.2
        
        # Follower ratio
        if profile.total_calls > 20 and profile.follower_count < 100:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def get_caller_recommendation(self, caller_id: str) -> Dict[str, Any]:
        """Get trading recommendation for caller"""
        
        if caller_id not in self.caller_profiles:
            return {
                'recommendation': 'AVOID',
                'confidence': 0.0,
                'reason': 'Unknown caller'
            }
        
        profile = self.caller_profiles[caller_id]
        
        # Determine recommendation
        if profile.manipulation_risk in [ManipulationRisk.HIGH, ManipulationRisk.EXTREME]:
            return {
                'recommendation': 'BLACKLIST',
                'confidence': 0.9,
                'reason': f'High manipulation risk: {profile.manipulation_risk.value}'
            }
        
        if profile.credibility_level == CredibilityLevel.ELITE:
            return {
                'recommendation': 'FOLLOW',
                'confidence': profile.trust_score,
                'reason': f'Elite caller with {profile.success_rate:.1%} success rate'
            }
        
        if profile.credibility_level == CredibilityLevel.HIGH:
            return {
                'recommendation': 'CONSIDER',
                'confidence': profile.trust_score,
                'reason': f'High credibility with {profile.success_rate:.1%} success rate'
            }
        
        if profile.total_calls < self.min_calls_for_rating:
            return {
                'recommendation': 'MONITOR',
                'confidence': 0.5,
                'reason': f'Insufficient data ({profile.total_calls} calls)'
            }
        
        return {
            'recommendation': 'CAUTION',
            'confidence': 0.3,
            'reason': f'Medium/low credibility: {profile.credibility_level.value}'
        }
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get intelligence summary"""
        
        total_callers = len(self.caller_profiles)
        elite_callers = sum(1 for p in self.caller_profiles.values() 
                           if p.credibility_level == CredibilityLevel.ELITE)
        high_risk_callers = sum(1 for p in self.caller_profiles.values()
                               if p.manipulation_risk in [ManipulationRisk.HIGH, ManipulationRisk.EXTREME])
        
        avg_success_rate = 0.0
        qualified_callers = [p for p in self.caller_profiles.values() if p.total_calls >= self.min_calls_for_rating]
        if qualified_callers:
            avg_success_rate = np.mean([p.success_rate for p in qualified_callers])
        
        return {
            'total_callers_tracked': total_callers,
            'elite_callers': elite_callers,
            'high_risk_callers': high_risk_callers,
            'total_calls_analyzed': len(self.call_analysis_cache),
            'avg_success_rate': avg_success_rate,
            'manipulation_detection_active': True
        }
    
    # Helper methods
    def _calculate_profile_completeness(self, caller_data: Dict[str, Any]) -> float:
        """Calculate profile completeness score"""
        fields = ['username', 'bio', 'location', 'website', 'profile_image']
        completed_fields = sum(1 for field in fields if caller_data.get(field))
        return completed_fields / len(fields)
    
    async def _extract_timeframe(self, text: str) -> Optional[str]:
        """Extract timeframe mentioned in call"""
        timeframe_patterns = [
            r'(\d+)\s*(minute|hour|day|week)s?',
            r'(short|medium|long)\s*term',
            r'(quick|fast|immediate)',
            r'(today|tomorrow|this week)'
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        return None
    
    async def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions"""
        return {
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': 'neutral',
            'volatility': 'medium',
            'volume': 'normal'
        }
    
    async def _update_caller_profile(self, profile: CallerProfile, caller_data: Dict[str, Any]):
        """Update existing caller profile"""
        profile.last_seen = datetime.now()
        profile.follower_count = caller_data.get('followers', profile.follower_count)
    
    async def _analyze_recent_activity(self, profile: CallerProfile):
        """Analyze recent activity patterns"""
        # Analyze posting frequency, content patterns, engagement metrics
        pass
    
    async def _save_caller_profile(self, profile: CallerProfile):
        """Save caller profile to database"""
        # Implementation would save to SQLite
        pass
    
    async def _save_call_analysis(self, analysis: CallAnalysis):
        """Save call analysis to database"""
        # Implementation would save to SQLite
        pass
    
    async def _load_existing_profiles(self):
        """Load existing profiles from database"""
        # Implementation would load from SQLite
        pass

def create_caller_intelligence_system(database_path: str = "caller_intelligence.db") -> AdvancedCallerIntelligence:
    """Create caller intelligence system"""
    return AdvancedCallerIntelligence(database_path)
