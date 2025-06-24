"""
Anti-Hype Filter System
========================

Ignores hyped tokens unless there's a genuine trigger (e.g., "LP live now").
Implements sophisticated hype detection and trigger validation.
"""

import asyncio
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp


class HypeLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TriggerType(Enum):
    LP_LIVE = "lp_live"
    LP_BURN = "lp_burn"
    MAJOR_LISTING = "major_listing"
    PARTNERSHIP = "partnership"
    PRODUCT_LAUNCH = "product_launch"
    AUDIT_COMPLETE = "audit_complete"
    CELEBRITY_ENDORSEMENT = "celebrity_endorsement"


@dataclass
class HypeAnalysis:
    """Analysis of token hype levels"""
    token_symbol: str
    hype_level: HypeLevel
    hype_score: float  # 0-1
    social_mentions: int
    influencer_mentions: int
    bot_activity_detected: bool
    organic_interest: float  # 0-1
    suspicious_patterns: List[str]


@dataclass
class TriggerEvent:
    """Legitimate trigger event detection"""
    trigger_type: TriggerType
    confidence: float  # 0-1
    source: str
    timestamp: datetime
    details: Dict
    verified: bool = False


@dataclass
class FilterResult:
    """Result of anti-hype filtering"""
    token_symbol: str
    passed_filter: bool
    hype_analysis: HypeAnalysis
    trigger_events: List[TriggerEvent]
    recommendation: str  # "PROCEED", "WAIT", "REJECT"
    reason: str


class AntiHypeFilter:
    """Filters out pure hype tokens while allowing legitimate triggers"""
    
    def __init__(self):
        self.logger = logging.getLogger("AntiHypeFilter")
        
        # Hype detection patterns
        self.hype_keywords = {
            "extreme": ["moon", "lambo", "1000x", "gem", "next doge", "pump it", "ape in"],
            "high": ["bullish", "diamond hands", "hodl", "to the moon", "rocket", "gains"],
            "medium": ["potential", "undervalued", "sleeper", "hidden gem", "breakout"]
        }
        
        # Legitimate trigger patterns
        self.trigger_patterns = {
            TriggerType.LP_LIVE: [
                r"LP\s+live\s+now",
                r"liquidity\s+pool\s+active",
                r"trading\s+live",
                r"LP\s+added",
                r"pool\s+created"
            ],
            TriggerType.LP_BURN: [
                r"LP\s+burn",
                r"liquidity\s+burned",
                r"LP\s+locked",
                r"liquidity\s+locked"
            ],
            TriggerType.MAJOR_LISTING: [
                r"listed\s+on\s+(binance|coinbase|kraken|huobi)",
                r"(binance|coinbase|kraken|huobi)\s+listing",
                r"tier\s+1\s+exchange"
            ],
            TriggerType.PARTNERSHIP: [
                r"partnership\s+with",
                r"collaboration\s+announcement",
                r"strategic\s+alliance"
            ],
            TriggerType.PRODUCT_LAUNCH: [
                r"mainnet\s+launch",
                r"platform\s+live",
                r"product\s+release",
                r"beta\s+launch"
            ]
        }
        
        # Known hype influencers and bots
        self.hype_accounts = set()
        self.bot_patterns = [
            r"follow\s+for\s+more",
            r"dm\s+for\s+alpha",
            r"not\s+financial\s+advice",
            r"dyor"
        ]
        
        # Session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Initialize the anti-hype filter"""
        
        self.session = aiohttp.ClientSession()
        self.logger.info("Anti-hype filter initialized")
        
    async def stop(self):
        """Clean shutdown"""
        
        if self.session:
            await self.session.close()
            
    async def analyze_token_hype(self, token_symbol: str) -> FilterResult:
        """Analyze token for hype vs legitimate triggers"""
        
        # Gather hype data
        hype_analysis = await self._analyze_hype_levels(token_symbol)
        
        # Detect trigger events
        trigger_events = await self._detect_trigger_events(token_symbol)
        
        # Apply filter logic
        passed_filter, recommendation, reason = self._apply_filter_logic(
            hype_analysis, trigger_events
        )
        
        result = FilterResult(
            token_symbol=token_symbol,
            passed_filter=passed_filter,
            hype_analysis=hype_analysis,
            trigger_events=trigger_events,
            recommendation=recommendation,
            reason=reason
        )
        
        self.logger.info(
            f"Anti-hype filter for {token_symbol}: {recommendation} "
            f"(hype: {hype_analysis.hype_level.value}, triggers: {len(trigger_events)})"
        )
        
        return result
        
    async def _analyze_hype_levels(self, token_symbol: str) -> HypeAnalysis:
        """Analyze social media hype levels"""
        
        # Initialize analysis
        analysis = HypeAnalysis(
            token_symbol=token_symbol,
            hype_level=HypeLevel.LOW,
            hype_score=0.0,
            social_mentions=0,
            influencer_mentions=0,
            bot_activity_detected=False,
            organic_interest=0.0,
            suspicious_patterns=[]
        )
        
        try:
            # Gather social media data (Twitter, Reddit, Telegram)
            social_data = await self._gather_social_data(token_symbol)
            
            # Analyze mention patterns
            analysis.social_mentions = social_data.get("total_mentions", 0)
            analysis.influencer_mentions = social_data.get("influencer_mentions", 0)
            
            # Detect bot activity
            analysis.bot_activity_detected = self._detect_bot_activity(social_data)
            
            # Calculate hype score
            analysis.hype_score = self._calculate_hype_score(social_data)
            
            # Determine hype level
            if analysis.hype_score > 0.8:
                analysis.hype_level = HypeLevel.EXTREME
            elif analysis.hype_score > 0.6:
                analysis.hype_level = HypeLevel.HIGH
            elif analysis.hype_score > 0.3:
                analysis.hype_level = HypeLevel.MEDIUM
            else:
                analysis.hype_level = HypeLevel.LOW
                
            # Calculate organic interest
            analysis.organic_interest = self._calculate_organic_interest(social_data)
            
            # Identify suspicious patterns
            analysis.suspicious_patterns = self._identify_suspicious_patterns(social_data)
            
        except Exception as e:
            self.logger.error(f"Error analyzing hype for {token_symbol}: {e}")
            
        return analysis
        
    async def _gather_social_data(self, token_symbol: str) -> Dict:
        """Gather social media data for analysis"""
        
        social_data = {
            "total_mentions": 0,
            "influencer_mentions": 0,
            "posts": [],
            "sentiment_scores": [],
            "time_distribution": {},
            "account_types": {"organic": 0, "bot": 0, "influencer": 0}
        }
        
        try:
            # Twitter data (mock implementation)
            twitter_data = await self._get_twitter_data(token_symbol)
            social_data.update(twitter_data)
            
            # Reddit data
            reddit_data = await self._get_reddit_data(token_symbol)
            self._merge_social_data(social_data, reddit_data)
            
            # Telegram monitoring
            telegram_data = await self._get_telegram_data(token_symbol)
            self._merge_social_data(social_data, telegram_data)
            
        except Exception as e:
            self.logger.error(f"Error gathering social data: {e}")
            
        return social_data
        
    async def _get_twitter_data(self, token_symbol: str) -> Dict:
        """Get Twitter mentions and analyze patterns"""
        
        # Mock implementation - in production, use Twitter API
        return {
            "total_mentions": 150,
            "influencer_mentions": 12,
            "posts": [
                {"text": f"${token_symbol} is going to moon!", "account_type": "bot"},
                {"text": f"LP live now for ${token_symbol}", "account_type": "organic"},
                {"text": f"1000x gem ${token_symbol}", "account_type": "bot"}
            ],
            "sentiment_scores": [0.8, 0.9, 0.95],
            "time_distribution": {"last_hour": 50, "last_6h": 100},
            "account_types": {"organic": 30, "bot": 80, "influencer": 12}
        }
        
    async def _get_reddit_data(self, token_symbol: str) -> Dict:
        """Get Reddit mentions and analyze patterns"""
        
        # Mock implementation
        return {
            "total_mentions": 25,
            "posts": [
                {"text": f"DD on ${token_symbol} - looks promising", "account_type": "organic"}
            ],
            "account_types": {"organic": 20, "bot": 5}
        }
        
    async def _get_telegram_data(self, token_symbol: str) -> Dict:
        """Get Telegram mentions and analyze patterns"""
        
        # Mock implementation
        return {
            "total_mentions": 80,
            "posts": [
                {"text": f"${token_symbol} LP burn confirmed!", "account_type": "organic"}
            ],
            "account_types": {"organic": 60, "bot": 20}
        }
        
    def _merge_social_data(self, main_data: Dict, new_data: Dict):
        """Merge social data from different sources"""
        
        main_data["total_mentions"] += new_data.get("total_mentions", 0)
        main_data["posts"].extend(new_data.get("posts", []))
        
        for account_type in ["organic", "bot", "influencer"]:
            main_data["account_types"][account_type] += new_data.get("account_types", {}).get(account_type, 0)
            
    def _detect_bot_activity(self, social_data: Dict) -> bool:
        """Detect bot activity patterns"""
        
        total_mentions = social_data.get("total_mentions", 0)
        bot_mentions = social_data.get("account_types", {}).get("bot", 0)
        
        if total_mentions == 0:
            return False
            
        bot_ratio = bot_mentions / total_mentions
        
        # High bot activity threshold
        if bot_ratio > 0.6:
            return True
            
        # Check for bot patterns in posts
        posts = social_data.get("posts", [])
        bot_pattern_count = 0
        
        for post in posts:
            text = post.get("text", "").lower()
            for pattern in self.bot_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    bot_pattern_count += 1
                    break
                    
        if len(posts) > 0 and bot_pattern_count / len(posts) > 0.5:
            return True
            
        return False
        
    def _calculate_hype_score(self, social_data: Dict) -> float:
        """Calculate overall hype score"""
        
        posts = social_data.get("posts", [])
        if not posts:
            return 0.0
            
        hype_score = 0.0
        total_posts = len(posts)
        
        for post in posts:
            text = post.get("text", "").lower()
            post_hype = 0.0
            
            # Check for extreme hype keywords
            for keyword in self.hype_keywords["extreme"]:
                if keyword in text:
                    post_hype += 0.3
                    
            # Check for high hype keywords
            for keyword in self.hype_keywords["high"]:
                if keyword in text:
                    post_hype += 0.2
                    
            # Check for medium hype keywords
            for keyword in self.hype_keywords["medium"]:
                if keyword in text:
                    post_hype += 0.1
                    
            hype_score += min(post_hype, 1.0)
            
        return min(hype_score / total_posts, 1.0)
        
    def _calculate_organic_interest(self, social_data: Dict) -> float:
        """Calculate organic vs artificial interest"""
        
        total_mentions = social_data.get("total_mentions", 0)
        organic_mentions = social_data.get("account_types", {}).get("organic", 0)
        
        if total_mentions == 0:
            return 0.0
            
        return organic_mentions / total_mentions
        
    def _identify_suspicious_patterns(self, social_data: Dict) -> List[str]:
        """Identify suspicious hype patterns"""
        
        patterns = []
        
        # High bot ratio
        total = social_data.get("total_mentions", 0)
        bots = social_data.get("account_types", {}).get("bot", 0)
        if total > 0 and bots / total > 0.6:
            patterns.append("High bot activity")
            
        # Sudden mention spike
        time_dist = social_data.get("time_distribution", {})
        recent = time_dist.get("last_hour", 0)
        if recent > 100:  # More than 100 mentions in last hour
            patterns.append("Sudden mention spike")
            
        # Repetitive content
        posts = social_data.get("posts", [])
        if len(posts) > 10:
            unique_texts = set(post.get("text", "") for post in posts)
            if len(unique_texts) / len(posts) < 0.3:  # Less than 30% unique content
                patterns.append("Repetitive content")
                
        return patterns
        
    async def _detect_trigger_events(self, token_symbol: str) -> List[TriggerEvent]:
        """Detect legitimate trigger events"""
        
        events = []
        
        try:
            # Gather announcement data
            announcements = await self._gather_announcements(token_symbol)
            
            # Check each trigger type
            for trigger_type, patterns in self.trigger_patterns.items():
                for announcement in announcements:
                    text = announcement.get("text", "").lower()
                    
                    for pattern in patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            event = TriggerEvent(
                                trigger_type=trigger_type,
                                confidence=0.8,  # Base confidence
                                source=announcement.get("source", "unknown"),
                                timestamp=datetime.now(),
                                details={"announcement": text},
                                verified=await self._verify_trigger(trigger_type, announcement)
                            )
                            events.append(event)
                            break
                            
        except Exception as e:
            self.logger.error(f"Error detecting triggers: {e}")
            
        return events
        
    async def _gather_announcements(self, token_symbol: str) -> List[Dict]:
        """Gather official announcements and news"""
        
        # Mock implementation - in production, gather from:
        # - Official token social accounts
        # - DEX announcements
        # - Partnership announcements
        # - Exchange listings
        
        return [
            {
                "text": f"LP live now for ${token_symbol} - trading active on Raydium",
                "source": "official_twitter",
                "timestamp": datetime.now(),
                "verified": True
            },
            {
                "text": f"${token_symbol} partnership with major DeFi protocol announced",
                "source": "partnership_announcement",
                "timestamp": datetime.now(),
                "verified": False
            }
        ]
        
    async def _verify_trigger(self, trigger_type: TriggerType, announcement: Dict) -> bool:
        """Verify legitimacy of trigger event"""
        
        # Mock verification - in production:
        # - Check on-chain data for LP events
        # - Verify exchange announcements
        # - Cross-reference with official sources
        
        source = announcement.get("source", "")
        
        if trigger_type == TriggerType.LP_LIVE:
            # Check on-chain for actual LP creation
            return True  # Mock verification
            
        elif trigger_type == TriggerType.MAJOR_LISTING:
            # Verify with exchange APIs
            return "official" in source
            
        return False
        
    def _apply_filter_logic(self, hype_analysis: HypeAnalysis, 
                          trigger_events: List[TriggerEvent]) -> Tuple[bool, str, str]:
        """Apply anti-hype filter logic"""
        
        # If extreme hype with no verified triggers - reject
        if hype_analysis.hype_level == HypeLevel.EXTREME and not any(e.verified for e in trigger_events):
            return False, "REJECT", "Extreme hype with no verified triggers"
            
        # If high hype with bot activity - reject
        if hype_analysis.hype_level == HypeLevel.HIGH and hype_analysis.bot_activity_detected:
            return False, "REJECT", "High hype with bot activity"
            
        # If verified LP live trigger - proceed regardless of hype
        lp_live_events = [e for e in trigger_events if e.trigger_type == TriggerType.LP_LIVE and e.verified]
        if lp_live_events:
            return True, "PROCEED", "Verified LP live trigger"
            
        # If verified major listing - proceed
        listing_events = [e for e in trigger_events if e.trigger_type == TriggerType.MAJOR_LISTING and e.verified]
        if listing_events:
            return True, "PROCEED", "Verified major exchange listing"
            
        # If medium/low hype with organic interest - wait for more confirmation
        if hype_analysis.hype_level in [HypeLevel.LOW, HypeLevel.MEDIUM] and hype_analysis.organic_interest > 0.6:
            return True, "WAIT", "Moderate hype with organic interest"
            
        # If low hype, minimal bot activity - proceed cautiously
        if hype_analysis.hype_level == HypeLevel.LOW and not hype_analysis.bot_activity_detected:
            return True, "PROCEED", "Low hype, organic interest"
            
        # Default: reject pure hype
        return False, "REJECT", f"Pure hype detected ({hype_analysis.hype_level.value})"


# Global instance
anti_hype_filter = AntiHypeFilter() 