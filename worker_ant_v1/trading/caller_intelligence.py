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
import sqlite3
import aiosqlite
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import aiosqlite
from worker_ant_v1.utils.logger import setup_logger

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
        self.logger = setup_logger("CallerIntelligence")
        
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
        self.logger.info("✅ Caller intelligence system initialized")
    
    async def process_social_event(self, event: Dict[str, Any]):
        """Process social media events for caller intelligence"""
        try:
            # Create caller data from social media event
            caller_data = {
                'caller_id': f"social_{event.get('user_id', 'unknown')}",
                'username': event.get('username', 'unknown'),
                'platform': Platform.TELEGRAM,  # Default platform 
                'account_age_days': event.get('account_age_days', 365),
                'follower_count': event.get('follower_count', 0),
                'verified_account': event.get('verified_account', False),
                'engagement_metrics': {
                    'likes': event.get('like_count', 0),
                    'shares': event.get('share_count', 0),
                    'replies': event.get('reply_count', 0)
                }
            }
            
            # Analyze the caller
            profile = await self.analyze_caller(caller_data)
            
            # If it's a social media post with crypto symbols, analyze the call
            event_type = event.get('event_type', '')
            mentioned_symbols = event.get('mentioned_symbols', [])
            
            if event_type in ['post', 'tweet', 'message'] and mentioned_symbols:
                call_data = {
                    'call_id': f"social_{event.get('id', 'unknown')}",
                    'caller_id': f"social_{event.get('user_id', 'unknown')}",
                    'token_symbols': mentioned_symbols,
                    'call_timestamp': event.get('timestamp', datetime.now()),
                    'platform': Platform.TELEGRAM,  # Default platform
                    'call_text': event.get('text', ''),
                    'engagement': {
                        'likes': event.get('like_count', 0),
                        'shares': event.get('share_count', 0),
                        'replies': event.get('reply_count', 0)
                    }
                }
                
                analysis = await self.analyze_call(call_data)
                
                # Log significant calls
                username = event.get('username', 'unknown')
                if analysis.manipulation_indicators or analysis.pump_dump_likelihood > 0.7:
                    self.logger.warning(f"🚨 High-risk social media call detected: {username} mentioned {mentioned_symbols}")
                elif mentioned_symbols:
                    self.logger.info(f"📊 Social media call tracked: {username} mentioned {mentioned_symbols}")
        
        except Exception as e:
            self.logger.error(f"Error processing social media event: {e}")
        
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
    
    async def _analyze_caller_credibility(self, caller_id: str, platform: Platform) -> Optional[Dict[str, Any]]:
        """Analyze caller credibility based on historical performance"""
        try:
            # Get caller's historical data
            caller_data = await self._get_caller_history(caller_id, platform)
            if not caller_data:
                return None
            
            # Calculate credibility metrics
            total_calls = len(caller_data['calls'])
            if total_calls < 5:  # Need minimum calls for analysis
                return {
                    'credibility_score': 0.3,
                    'confidence': 0.2,
                    'reason': 'Insufficient call history'
                }
            
            # Calculate success rate
            successful_calls = sum(1 for call in caller_data['calls'] if call.get('success', False))
            success_rate = successful_calls / total_calls
            
            # Calculate average profit
            profits = [call.get('profit_percent', 0) for call in caller_data['calls']]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # Calculate consistency (standard deviation of profits)
            profit_std = np.std(profits) if len(profits) > 1 else 0
            consistency_score = max(0, 1 - (profit_std / 10))  # Lower std = higher consistency
            
            # Calculate credibility score
            credibility_score = (
                success_rate * 0.4 +
                min(avg_profit / 10, 1.0) * 0.3 +
                consistency_score * 0.2 +
                min(total_calls / 50, 1.0) * 0.1  # Bonus for experience
            )
            
            # Determine confidence level
            if total_calls >= 20 and success_rate >= 0.6:
                confidence = 0.9
            elif total_calls >= 10 and success_rate >= 0.5:
                confidence = 0.7
            else:
                confidence = 0.5
            
            return {
                'credibility_score': round(credibility_score, 3),
                'confidence': round(confidence, 3),
                'success_rate': round(success_rate, 3),
                'avg_profit': round(avg_profit, 2),
                'total_calls': total_calls,
                'consistency_score': round(consistency_score, 3),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing caller credibility: {e}")
            return None
    
    async def _analyze_recent_activity(self, profile: CallerProfile):
        """Analyze recent activity patterns"""
        try:
            # Get recent calls for this caller
            recent_calls = [call for call in self.call_analysis_cache 
                          if hasattr(call, 'caller_id') and call.caller_id == profile.caller_id]
            
            if not recent_calls:
                return
            
            # Analyze posting frequency
            call_times = [call.call_timestamp for call in recent_calls if hasattr(call, 'call_timestamp')]
            if len(call_times) > 1:
                time_diffs = []
                for i in range(1, len(call_times)):
                    diff = (call_times[i] - call_times[i-1]).total_seconds() / 60  # minutes
                    time_diffs.append(diff)
                
                avg_time_between_calls = sum(time_diffs) / len(time_diffs)
                
                # Detect unusual patterns
                if avg_time_between_calls < 5:  # Less than 5 minutes between calls
                    profile.risk_indicators.append("high_frequency_posting")
                elif avg_time_between_calls > 1440:  # More than 24 hours
                    profile.risk_indicators.append("inactive_period")
            
            # Analyze content patterns
            call_texts = [call.call_text for call in recent_calls if hasattr(call, 'call_text')]
            if call_texts:
                # Check for repetitive content
                unique_texts = set(call_texts)
                if len(unique_texts) / len(call_texts) < 0.7:  # Less than 70% unique content
                    profile.risk_indicators.append("repetitive_content")
                
                # Check for copy-paste behavior
                for i, text1 in enumerate(call_texts):
                    for j, text2 in enumerate(call_texts[i+1:], i+1):
                        if text1 == text2:
                            profile.risk_indicators.append("copy_paste_behavior")
                            break
            
            # Analyze engagement patterns
            if hasattr(profile, 'engagement_rate'):
                if profile.engagement_rate < 0.01:  # Less than 1% engagement
                    profile.risk_indicators.append("low_engagement")
                elif profile.engagement_rate > 0.5:  # More than 50% engagement (suspicious)
                    profile.risk_indicators.append("suspicious_engagement")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing recent activity: {e}")
    
    async def _save_caller_profile(self, profile: CallerProfile):
        """Save caller profile to database"""
        try:
            # Save to SQLite database
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO caller_profiles (
                    caller_id, username, platform, first_seen, last_seen,
                    account_age_days, follower_count, verified_account, profile_completeness,
                    total_calls, successful_calls, failed_calls, success_rate,
                    avg_profit_percent, avg_time_to_profit_minutes,
                    credibility_level, manipulation_risk, trust_score,
                    engagement_rate, sentiment_alignment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.caller_id, profile.username, profile.platform.value,
                profile.first_seen.isoformat(), profile.last_seen.isoformat(),
                profile.account_age_days, profile.follower_count, profile.verified_account,
                profile.profile_completeness, profile.total_calls, profile.successful_calls,
                profile.failed_calls, profile.success_rate, profile.avg_profit_percent,
                profile.avg_time_to_profit_minutes, profile.credibility_level.value,
                profile.manipulation_risk.value, profile.trust_score,
                profile.engagement_rate, profile.sentiment_alignment
            ))
            
            # Save risk indicators
            for indicator in profile.risk_indicators:
                cursor.execute("""
                    INSERT OR IGNORE INTO caller_risk_indicators (caller_id, indicator)
                    VALUES (?, ?)
                """, (profile.caller_id, indicator))
            
            # Save network connections
            for connection in profile.network_connections:
                cursor.execute("""
                    INSERT OR IGNORE INTO caller_connections (caller_id, connected_to)
                    VALUES (?, ?)
                """, (profile.caller_id, connection))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving caller profile: {e}")
    
    async def _save_call_analysis(self, analysis: CallAnalysis):
        """Save call analysis to database"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO call_analysis (
                    call_id, caller_id, token_address, token_symbol, call_timestamp,
                    platform, call_text, confidence_language, timeframe_mentioned,
                    token_age_hours, liquidity_at_call, holder_count_at_call,
                    outcome_tracked, max_profit_achieved, time_to_peak_minutes,
                    final_outcome, pump_dump_likelihood
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.call_id, analysis.caller_id, analysis.token_address,
                analysis.token_symbol, analysis.call_timestamp.isoformat(),
                analysis.platform.value, analysis.call_text, analysis.confidence_language,
                analysis.timeframe_mentioned, analysis.token_age_hours,
                analysis.liquidity_at_call, analysis.holder_count_at_call,
                analysis.outcome_tracked, analysis.max_profit_achieved,
                analysis.time_to_peak_minutes, analysis.final_outcome,
                analysis.pump_dump_likelihood
            ))
            
            # Save urgency indicators
            for indicator in analysis.urgency_indicators:
                cursor.execute("""
                    INSERT OR IGNORE INTO call_urgency_indicators (call_id, indicator)
                    VALUES (?, ?)
                """, (analysis.call_id, indicator))
            
            # Save price predictions
            for prediction in analysis.price_predictions:
                cursor.execute("""
                    INSERT OR IGNORE INTO call_price_predictions (call_id, prediction)
                    VALUES (?, ?)
                """, (analysis.call_id, prediction))
            
            # Save manipulation indicators
            for indicator in analysis.manipulation_indicators:
                cursor.execute("""
                    INSERT OR IGNORE INTO call_manipulation_indicators (call_id, indicator)
                    VALUES (?, ?)
                """, (analysis.call_id, indicator))
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving call analysis: {e}")
    
    async def _load_existing_profiles(self):
        """Load existing profiles from database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Load caller profiles
            cursor.execute("""
                SELECT * FROM caller_profiles
            """)
            
            rows = cursor.fetchall()
            for row in rows:
                # Create CallerProfile from database row
                profile = CallerProfile(
                    caller_id=row[0],
                    username=row[1],
                    platform=Platform(row[2]),
                    first_seen=datetime.fromisoformat(row[3]),
                    last_seen=datetime.fromisoformat(row[4]),
                    account_age_days=row[5],
                    follower_count=row[6],
                    verified_account=bool(row[7]),
                    profile_completeness=row[8],
                    total_calls=row[9],
                    successful_calls=row[10],
                    failed_calls=row[11],
                    success_rate=row[12],
                    avg_profit_percent=row[13],
                    avg_time_to_profit_minutes=row[14],
                    credibility_level=CredibilityLevel(row[15]),
                    manipulation_risk=ManipulationRisk(row[16]),
                    trust_score=row[17],
                    risk_indicators=[],
                    engagement_rate=row[18],
                    sentiment_alignment=row[19],
                    network_connections=[],
                    associated_groups=[],
                    recent_calls=[],
                    performance_history=[]
                )
                
                # Load risk indicators
                cursor.execute("""
                    SELECT indicator FROM caller_risk_indicators WHERE caller_id = ?
                """, (profile.caller_id,))
                
                risk_indicators = cursor.fetchall()
                profile.risk_indicators = [indicator[0] for indicator in risk_indicators]
                
                # Load network connections
                cursor.execute("""
                    SELECT connected_to FROM caller_connections WHERE caller_id = ?
                """, (profile.caller_id,))
                
                connections = cursor.fetchall()
                profile.network_connections = [conn[0] for conn in connections]
                
                self.caller_profiles[profile.caller_id] = profile
            
            # Load recent call analysis
            cursor.execute("""
                SELECT * FROM call_analysis 
                WHERE call_timestamp > datetime('now', '-7 days')
                ORDER BY call_timestamp DESC
            """)
            
            call_rows = cursor.fetchall()
            for row in call_rows:
                analysis = CallAnalysis(
                    call_id=row[0],
                    caller_id=row[1],
                    token_address=row[2],
                    token_symbol=row[3],
                    call_timestamp=datetime.fromisoformat(row[4]),
                    platform=Platform(row[5]),
                    call_text=row[6],
                    confidence_language=row[7],
                    urgency_indicators=[],
                    price_predictions=[],
                    timeframe_mentioned=row[8],
                    market_conditions={},
                    token_age_hours=row[9],
                    liquidity_at_call=row[10],
                    holder_count_at_call=row[11],
                    outcome_tracked=bool(row[12]),
                    max_profit_achieved=row[13],
                    time_to_peak_minutes=row[14],
                    final_outcome=row[15],
                    manipulation_indicators=[],
                    coordination_signals=[],
                    pump_dump_likelihood=row[16]
                )
                
                # Load urgency indicators
                cursor.execute("""
                    SELECT indicator FROM call_urgency_indicators WHERE call_id = ?
                """, (analysis.call_id,))
                
                urgency_indicators = cursor.fetchall()
                analysis.urgency_indicators = [indicator[0] for indicator in urgency_indicators]
                
                # Load price predictions
                cursor.execute("""
                    SELECT prediction FROM call_price_predictions WHERE call_id = ?
                """, (analysis.call_id,))
                
                price_predictions = cursor.fetchall()
                analysis.price_predictions = [pred[0] for pred in price_predictions]
                
                # Load manipulation indicators
                cursor.execute("""
                    SELECT indicator FROM call_manipulation_indicators WHERE call_id = ?
                """, (analysis.call_id,))
                
                manipulation_indicators = cursor.fetchall()
                analysis.manipulation_indicators = [indicator[0] for indicator in manipulation_indicators]
                
                self.call_analysis_cache.append(analysis)
            
            self.logger.info(f"Loaded {len(self.caller_profiles)} caller profiles and {len(self.call_analysis_cache)} call analyses")
            
        except Exception as e:
            self.logger.error(f"Error loading existing profiles: {e}")

    async def get_recent_calls(self, source_id: str, token_address: str, hours: int = 6) -> List[Dict[str, Any]]:
        """Get recent calls from a specific source about a token"""
        
        try:
            # Query the actual database
            async with aiosqlite.connect(self.database_path) as db:
                query = """
                SELECT * FROM call_analysis 
                WHERE caller_id = ? AND token_address = ? 
                AND call_timestamp > datetime('now', '-{} hours')
                ORDER BY call_timestamp DESC
                """.format(hours)
                
                cursor = await db.execute(query, (source_id, token_address))
                rows = await cursor.fetchall()
                
                recent_calls = []
                for row in rows:
                    recent_calls.append({
                        'call_id': row[0],
                        'caller_id': row[1],
                        'token_address': row[2],
                        'token_symbol': row[3],
                        'call_timestamp': datetime.fromisoformat(row[4]),
                        'call_text': row[6],
                        'confidence_language': row[7],
                        'sentiment_score': getattr(row, 'sentiment_score', 0.0),
                        'platform': row[5]
                    })
                
                return recent_calls
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get recent calls for {source_id}: {e}")
            return []

    async def get_token_calls(self, token_address: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get all recent calls about a specific token"""
        
        try:
            all_calls = []
            
            # Search through recent analysis cache
            for analysis in self.call_analysis_cache:
                if (hasattr(analysis, 'token_address') and analysis.token_address == token_address):
                    call_time = analysis.call_timestamp if hasattr(analysis, 'call_timestamp') else datetime.now()
                    if call_time > datetime.now() - timedelta(hours=hours):
                        all_calls.append({
                            'call_id': analysis.call_id if hasattr(analysis, 'call_id') else f"call_{int(time.time())}",
                            'caller_id': analysis.caller_id if hasattr(analysis, 'caller_id') else 'unknown',
                            'token_address': analysis.token_address,
                            'token_symbol': analysis.token_symbol if hasattr(analysis, 'token_symbol') else 'UNKNOWN',
                            'call_timestamp': call_time,
                            'call_text': analysis.call_text if hasattr(analysis, 'call_text') else '',
                            'confidence_language': analysis.confidence_language if hasattr(analysis, 'confidence_language') else 0.5,
                            'sentiment_score': getattr(analysis, 'sentiment_score', 0.0),
                            'platform': analysis.platform.value if hasattr(analysis, 'platform') else 'twitter'
                        })
            
            # Sort by timestamp
            all_calls.sort(key=lambda x: x['call_timestamp'], reverse=True)
            
            return all_calls
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get token calls for {token_address}: {e}")
            return []

    async def analyze_twitter_mentions(self, token_symbol: str, token_address: str) -> Dict[str, Any]:
        """Analyze Twitter mentions for a token (integration point for Twitter sentiment)"""
        
        try:
            # This will be called by the Twitter sentiment analyzer
            # For now, return basic structure
            
            mentions_data = {
                'token_symbol': token_symbol,
                'token_address': token_address,
                'total_mentions': 0,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'top_callers': [],
                'analyzed_at': datetime.now()
            }
            
            # Check if we have any recent calls for this token
            recent_calls = await self.get_token_calls(token_address, hours=6)
            
            if recent_calls:
                mentions_data['total_mentions'] = len(recent_calls)
                
                # Calculate average sentiment
                sentiments = [call.get('sentiment_score', 0.0) for call in recent_calls]
                if sentiments:
                    mentions_data['sentiment_score'] = sum(sentiments) / len(sentiments)
                
                # Get top callers
                caller_counts = {}
                for call in recent_calls:
                    caller_id = call.get('caller_id', 'unknown')
                    caller_counts[caller_id] = caller_counts.get(caller_id, 0) + 1
                
                mentions_data['top_callers'] = sorted(
                    caller_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            
            return mentions_data
            
        except Exception as e:
            self.logger.error(f"❌ Twitter mentions analysis failed: {e}")
            return {
                'token_symbol': token_symbol,
                'token_address': token_address,
                'total_mentions': 0,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'top_callers': [],
                'analyzed_at': datetime.now()
            }

def create_caller_intelligence_system(database_path: str = "caller_intelligence.db") -> AdvancedCallerIntelligence:
    """Create caller intelligence system"""
    return AdvancedCallerIntelligence(database_path)
