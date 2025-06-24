"""
Caller Credibility Tracking System
==================================

Maintains a database of Telegram/Discord callers and ranks them by historical quality.
Tracks win rates, signal accuracy, and overall credibility for better trade decisions.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import aiofiles


class CallerType(Enum):
    ALPHA_CALLER = "alpha_caller"
    INFLUENCER = "influencer"
    DEGEN = "degen"
    BOT = "bot"
    VERIFIED = "verified"
    SCAMMER = "scammer"


class SignalQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    SCAM = "scam"


@dataclass
class CallerProfile:
    """Profile of a signal caller"""
    caller_id: str
    username: str
    platform: str  # telegram, discord, twitter
    caller_type: CallerType
    
    # Performance metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    scam_calls: int = 0
    win_rate: float = 0.0
    
    # Quality metrics
    avg_profit_percent: float = 0.0
    avg_time_to_profit: int = 0  # minutes
    consistency_score: float = 0.0  # 0-1
    
    # Trust metrics
    credibility_score: float = 0.5  # 0-1, starts at neutral
    trust_level: str = "unknown"
    verification_status: bool = False
    
    # Historical data
    first_seen: datetime = None
    last_seen: datetime = None
    call_history: List[Dict] = None
    
    # Social metrics
    follower_count: int = 0
    engagement_rate: float = 0.0
    
    def __post_init__(self):
        if self.call_history is None:
            self.call_history = []
        if self.first_seen is None:
            self.first_seen = datetime.now()


@dataclass
class CallSignal:
    """Individual call signal from a caller"""
    signal_id: str
    caller_id: str
    token_symbol: str
    token_address: str
    call_type: str  # "buy", "sell", "hold"
    
    # Signal details
    timestamp: datetime
    platform: str
    message_text: str
    confidence_level: str  # "high", "medium", "low"
    
    # Outcome tracking
    outcome_tracked: bool = False
    was_successful: bool = False
    profit_percent: float = 0.0
    time_to_outcome: int = 0  # minutes
    
    # Signal quality assessment
    signal_quality: SignalQuality = SignalQuality.AVERAGE
    rug_potential: float = 0.0  # 0-1


@dataclass
class CredibilityUpdate:
    """Update to caller credibility based on performance"""
    caller_id: str
    old_score: float
    new_score: float
    reason: str
    impact: float
    timestamp: datetime


class CallerCredibilityTracker:
    """Tracks and ranks caller credibility across platforms"""
    
    def __init__(self, db_path: str = "worker_ant_v1/caller_credibility.db"):
        self.logger = logging.getLogger("CallerCredibility")
        self.db_path = db_path
        self.caller_profiles: Dict[str, CallerProfile] = {}
        self.call_signals: Dict[str, CallSignal] = {}
        
        # Credibility scoring weights
        self.scoring_weights = {
            "win_rate": 0.3,
            "avg_profit": 0.25,
            "consistency": 0.2,
            "call_frequency": 0.1,
            "verification": 0.1,
            "social_proof": 0.05
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            "excellent_caller": {"win_rate": 0.75, "avg_profit": 15.0, "min_calls": 20},
            "good_caller": {"win_rate": 0.65, "avg_profit": 10.0, "min_calls": 10},
            "average_caller": {"win_rate": 0.5, "avg_profit": 5.0, "min_calls": 5},
            "poor_caller": {"win_rate": 0.3, "avg_profit": 0.0, "min_calls": 3}
        }
        
    async def initialize(self):
        """Initialize the credibility tracking system"""
        
        await self._setup_database()
        await self._load_existing_data()
        
        self.logger.info("Caller credibility tracker initialized")
        
    async def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Caller profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS caller_profiles (
                caller_id TEXT PRIMARY KEY,
                username TEXT,
                platform TEXT,
                caller_type TEXT,
                total_calls INTEGER DEFAULT 0,
                successful_calls INTEGER DEFAULT 0,
                failed_calls INTEGER DEFAULT 0,
                scam_calls INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0.0,
                avg_profit_percent REAL DEFAULT 0.0,
                avg_time_to_profit INTEGER DEFAULT 0,
                consistency_score REAL DEFAULT 0.0,
                credibility_score REAL DEFAULT 0.5,
                trust_level TEXT DEFAULT 'unknown',
                verification_status BOOLEAN DEFAULT FALSE,
                first_seen TEXT,
                last_seen TEXT,
                follower_count INTEGER DEFAULT 0,
                engagement_rate REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Call signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS call_signals (
                signal_id TEXT PRIMARY KEY,
                caller_id TEXT,
                token_symbol TEXT,
                token_address TEXT,
                call_type TEXT,
                timestamp TEXT,
                platform TEXT,
                message_text TEXT,
                confidence_level TEXT,
                outcome_tracked BOOLEAN DEFAULT FALSE,
                was_successful BOOLEAN DEFAULT FALSE,
                profit_percent REAL DEFAULT 0.0,
                time_to_outcome INTEGER DEFAULT 0,
                signal_quality TEXT DEFAULT 'average',
                rug_potential REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (caller_id) REFERENCES caller_profiles (caller_id)
            )
        ''')
        
        # Credibility updates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS credibility_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                caller_id TEXT,
                old_score REAL,
                new_score REAL,
                reason TEXT,
                impact REAL,
                timestamp TEXT,
                FOREIGN KEY (caller_id) REFERENCES caller_profiles (caller_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def _load_existing_data(self):
        """Load existing caller data from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load caller profiles
        cursor.execute("SELECT * FROM caller_profiles")
        for row in cursor.fetchall():
            profile = CallerProfile(
                caller_id=row[0],
                username=row[1],
                platform=row[2],
                caller_type=CallerType(row[3]),
                total_calls=row[4],
                successful_calls=row[5],
                failed_calls=row[6],
                scam_calls=row[7],
                win_rate=row[8],
                avg_profit_percent=row[9],
                avg_time_to_profit=row[10],
                consistency_score=row[11],
                credibility_score=row[12],
                trust_level=row[13],
                verification_status=bool(row[14]),
                first_seen=datetime.fromisoformat(row[15]) if row[15] else datetime.now(),
                last_seen=datetime.fromisoformat(row[16]) if row[16] else datetime.now(),
                follower_count=row[17],
                engagement_rate=row[18]
            )
            self.caller_profiles[profile.caller_id] = profile
            
        conn.close()
        self.logger.info(f"Loaded {len(self.caller_profiles)} caller profiles")
        
    async def register_caller(self, username: str, platform: str, 
                            caller_type: CallerType = CallerType.ALPHA_CALLER,
                            follower_count: int = 0) -> CallerProfile:
        """Register a new caller"""
        
        caller_id = f"{platform}_{username}"
        
        if caller_id in self.caller_profiles:
            return self.caller_profiles[caller_id]
            
        profile = CallerProfile(
            caller_id=caller_id,
            username=username,
            platform=platform,
            caller_type=caller_type,
            follower_count=follower_count,
            first_seen=datetime.now(),
            last_seen=datetime.now()
        )
        
        self.caller_profiles[caller_id] = profile
        await self._save_caller_profile(profile)
        
        self.logger.info(f"Registered new caller: {username} on {platform}")
        return profile
        
    async def record_call_signal(self, caller_id: str, token_symbol: str, 
                               token_address: str, call_type: str,
                               message_text: str, confidence_level: str = "medium") -> CallSignal:
        """Record a new call signal from a caller"""
        
        signal_id = f"{caller_id}_{token_symbol}_{int(datetime.now().timestamp())}"
        
        # Get caller profile
        if caller_id not in self.caller_profiles:
            # Auto-register unknown caller
            username = caller_id.split("_", 1)[1] if "_" in caller_id else caller_id
            platform = caller_id.split("_")[0] if "_" in caller_id else "unknown"
            await self.register_caller(username, platform)
            
        caller = self.caller_profiles[caller_id]
        
        signal = CallSignal(
            signal_id=signal_id,
            caller_id=caller_id,
            token_symbol=token_symbol,
            token_address=token_address,
            call_type=call_type,
            timestamp=datetime.now(),
            platform=caller.platform,
            message_text=message_text,
            confidence_level=confidence_level
        )
        
        self.call_signals[signal_id] = signal
        caller.call_history.append(asdict(signal))
        caller.total_calls += 1
        caller.last_seen = datetime.now()
        
        await self._save_call_signal(signal)
        await self._save_caller_profile(caller)
        
        self.logger.info(f"Recorded call signal: {caller_id} -> {token_symbol} ({call_type})")
        return signal
        
    async def update_signal_outcome(self, signal_id: str, was_successful: bool, 
                                  profit_percent: float, time_to_outcome: int = 0):
        """Update the outcome of a call signal"""
        
        if signal_id not in self.call_signals:
            self.logger.warning(f"Signal not found: {signal_id}")
            return
            
        signal = self.call_signals[signal_id]
        caller = self.caller_profiles[signal.caller_id]
        
        # Update signal
        signal.outcome_tracked = True
        signal.was_successful = was_successful
        signal.profit_percent = profit_percent
        signal.time_to_outcome = time_to_outcome
        
        # Assess signal quality
        signal.signal_quality = self._assess_signal_quality(profit_percent, time_to_outcome)
        
        # Update caller metrics
        if was_successful:
            caller.successful_calls += 1
        else:
            caller.failed_calls += 1
            
        # Detect potential rug/scam
        if profit_percent < -80:  # Lost >80% - likely rug
            caller.scam_calls += 1
            signal.rug_potential = 1.0
            
        # Recalculate caller metrics
        await self._recalculate_caller_metrics(caller)
        
        # Update credibility score
        await self._update_credibility_score(caller, signal)
        
        await self._save_call_signal(signal)
        await self._save_caller_profile(caller)
        
        self.logger.info(
            f"Updated signal outcome: {signal_id} -> {'SUCCESS' if was_successful else 'FAIL'} "
            f"({profit_percent:.1f}%)"
        )
        
    def _assess_signal_quality(self, profit_percent: float, time_to_outcome: int) -> SignalQuality:
        """Assess the quality of a signal based on outcomes"""
        
        if profit_percent < -80:
            return SignalQuality.SCAM
        elif profit_percent < -20:
            return SignalQuality.POOR
        elif profit_percent < 5:
            return SignalQuality.AVERAGE
        elif profit_percent < 20:
            return SignalQuality.GOOD
        else:
            return SignalQuality.EXCELLENT
            
    async def _recalculate_caller_metrics(self, caller: CallerProfile):
        """Recalculate all metrics for a caller"""
        
        if caller.total_calls == 0:
            return
            
        # Win rate
        caller.win_rate = caller.successful_calls / caller.total_calls
        
        # Average profit (only from successful calls)
        if caller.successful_calls > 0:
            total_profit = sum(
                call.get("profit_percent", 0) 
                for call in caller.call_history 
                if call.get("was_successful", False)
            )
            caller.avg_profit_percent = total_profit / caller.successful_calls
        
        # Average time to profit
        if caller.successful_calls > 0:
            total_time = sum(
                call.get("time_to_outcome", 0)
                for call in caller.call_history
                if call.get("was_successful", False)
            )
            caller.avg_time_to_profit = total_time // caller.successful_calls
            
        # Consistency score (variance in performance)
        if len(caller.call_history) >= 5:
            profits = [call.get("profit_percent", 0) for call in caller.call_history]
            avg_profit = sum(profits) / len(profits)
            variance = sum((p - avg_profit) ** 2 for p in profits) / len(profits)
            # Lower variance = higher consistency
            caller.consistency_score = max(0, 1 - (variance / 1000))  # Normalize
            
    async def _update_credibility_score(self, caller: CallerProfile, latest_signal: CallSignal):
        """Update caller's credibility score based on latest performance"""
        
        old_score = caller.credibility_score
        
        # Base score calculation using weighted metrics
        win_rate_score = caller.win_rate * self.scoring_weights["win_rate"]
        profit_score = min(caller.avg_profit_percent / 50, 1.0) * self.scoring_weights["avg_profit"]
        consistency_score = caller.consistency_score * self.scoring_weights["consistency"]
        
        # Frequency bonus (more calls = more reliable data)
        frequency_score = min(caller.total_calls / 50, 1.0) * self.scoring_weights["call_frequency"]
        
        # Verification bonus
        verification_score = (1.0 if caller.verification_status else 0.0) * self.scoring_weights["verification"]
        
        # Social proof (follower count, engagement)
        social_score = min(caller.follower_count / 10000, 1.0) * self.scoring_weights["social_proof"]
        
        new_score = (
            win_rate_score + profit_score + consistency_score + 
            frequency_score + verification_score + social_score
        )
        
        # Penalties
        if caller.scam_calls > 0:
            scam_penalty = (caller.scam_calls / caller.total_calls) * 0.5
            new_score = max(0, new_score - scam_penalty)
            
        # Smooth score updates (avoid wild swings)
        if caller.total_calls < 10:
            # Fast updates for new callers
            caller.credibility_score = new_score
        else:
            # Gradual updates for established callers
            caller.credibility_score = (old_score * 0.8) + (new_score * 0.2)
            
        # Update trust level
        caller.trust_level = self._calculate_trust_level(caller)
        
        # Record credibility update
        update = CredibilityUpdate(
            caller_id=caller.caller_id,
            old_score=old_score,
            new_score=caller.credibility_score,
            reason=f"Signal outcome: {latest_signal.signal_quality.value}",
            impact=abs(caller.credibility_score - old_score),
            timestamp=datetime.now()
        )
        
        await self._save_credibility_update(update)
        
    def _calculate_trust_level(self, caller: CallerProfile) -> str:
        """Calculate trust level based on credibility score and metrics"""
        
        score = caller.credibility_score
        
        # Check for excellent caller criteria
        excellent = self.quality_thresholds["excellent_caller"]
        if (caller.win_rate >= excellent["win_rate"] and 
            caller.avg_profit_percent >= excellent["avg_profit"] and
            caller.total_calls >= excellent["min_calls"] and
            caller.scam_calls == 0):
            return "trusted_alpha"
            
        # Check for good caller criteria
        good = self.quality_thresholds["good_caller"]
        if (caller.win_rate >= good["win_rate"] and 
            caller.avg_profit_percent >= good["avg_profit"] and
            caller.total_calls >= good["min_calls"]):
            return "reliable"
            
        # Check for average caller
        average = self.quality_thresholds["average_caller"]
        if (caller.win_rate >= average["win_rate"] and 
            caller.total_calls >= average["min_calls"]):
            return "moderate"
            
        # Check for poor performance
        if caller.scam_calls > 0 or caller.win_rate < 0.3:
            return "unreliable"
            
        return "unknown"
        
    async def get_caller_ranking(self, limit: int = 50) -> List[CallerProfile]:
        """Get top callers ranked by credibility"""
        
        # Filter callers with minimum activity
        active_callers = [
            caller for caller in self.caller_profiles.values()
            if caller.total_calls >= 5  # Minimum 5 calls
        ]
        
        # Sort by credibility score
        ranked_callers = sorted(
            active_callers,
            key=lambda c: (c.credibility_score, c.win_rate, c.total_calls),
            reverse=True
        )
        
        return ranked_callers[:limit]
        
    async def get_caller_recommendations(self, token_symbol: str) -> List[Tuple[CallerProfile, float]]:
        """Get caller recommendations for a specific token"""
        
        recommendations = []
        
        for caller in self.caller_profiles.values():
            if caller.total_calls < 3:  # Skip callers with too few calls
                continue
                
            # Check if caller has called this token before
            token_calls = [
                call for call in caller.call_history
                if call.get("token_symbol") == token_symbol
            ]
            
            # Base recommendation score
            rec_score = caller.credibility_score
            
            # Bonus for token-specific success
            if token_calls:
                token_successes = sum(1 for call in token_calls if call.get("was_successful", False))
                token_win_rate = token_successes / len(token_calls)
                rec_score *= (1 + token_win_rate * 0.3)  # Up to 30% bonus
                
            # Penalty for recent failures
            recent_calls = [
                call for call in caller.call_history[-10:]  # Last 10 calls
                if not call.get("was_successful", True)
            ]
            if len(recent_calls) > 3:  # More than 3 recent failures
                rec_score *= 0.7
                
            recommendations.append((caller, rec_score))
            
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
        
    async def _save_caller_profile(self, caller: CallerProfile):
        """Save caller profile to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO caller_profiles 
            (caller_id, username, platform, caller_type, total_calls, successful_calls,
             failed_calls, scam_calls, win_rate, avg_profit_percent, avg_time_to_profit,
             consistency_score, credibility_score, trust_level, verification_status,
             first_seen, last_seen, follower_count, engagement_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            caller.caller_id, caller.username, caller.platform, caller.caller_type.value,
            caller.total_calls, caller.successful_calls, caller.failed_calls, caller.scam_calls,
            caller.win_rate, caller.avg_profit_percent, caller.avg_time_to_profit,
            caller.consistency_score, caller.credibility_score, caller.trust_level,
            caller.verification_status, caller.first_seen.isoformat(), caller.last_seen.isoformat(),
            caller.follower_count, caller.engagement_rate
        ))
        
        conn.commit()
        conn.close()
        
    async def _save_call_signal(self, signal: CallSignal):
        """Save call signal to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO call_signals
            (signal_id, caller_id, token_symbol, token_address, call_type, timestamp,
             platform, message_text, confidence_level, outcome_tracked, was_successful,
             profit_percent, time_to_outcome, signal_quality, rug_potential)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id, signal.caller_id, signal.token_symbol, signal.token_address,
            signal.call_type, signal.timestamp.isoformat(), signal.platform, signal.message_text,
            signal.confidence_level, signal.outcome_tracked, signal.was_successful,
            signal.profit_percent, signal.time_to_outcome, signal.signal_quality.value,
            signal.rug_potential
        ))
        
        conn.commit()
        conn.close()
        
    async def _save_credibility_update(self, update: CredibilityUpdate):
        """Save credibility update to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO credibility_updates
            (caller_id, old_score, new_score, reason, impact, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            update.caller_id, update.old_score, update.new_score,
            update.reason, update.impact, update.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_credibility_summary(self) -> Dict:
        """Get summary of credibility tracking system"""
        
        total_callers = len(self.caller_profiles)
        trusted_callers = len([c for c in self.caller_profiles.values() if c.trust_level == "trusted_alpha"])
        reliable_callers = len([c for c in self.caller_profiles.values() if c.trust_level == "reliable"])
        
        return {
            "total_callers": total_callers,
            "trusted_callers": trusted_callers,
            "reliable_callers": reliable_callers,
            "total_signals": len(self.call_signals),
            "avg_credibility": sum(c.credibility_score for c in self.caller_profiles.values()) / total_callers if total_callers > 0 else 0
        }


# Global instance
caller_credibility_tracker = CallerCredibilityTracker() 