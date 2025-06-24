"""
Rug Memory System
=================

Skips trades with patterns resembling past scams.
Implements pattern recognition and scam detection based on historical data.
"""

import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class RugType(Enum):
    HONEYPOT = "honeypot"
    LP_DRAIN = "lp_drain"
    MINT_DISABLE = "mint_disable"
    WHALE_DUMP = "whale_dump"
    DEV_DUMP = "dev_dump"
    TAX_CHANGE = "tax_change"
    BLACKLIST_ATTACK = "blacklist_attack"
    PAUSE_ATTACK = "pause_attack"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RugPattern:
    """Pattern of a rug pull or scam"""
    pattern_id: str
    rug_type: RugType
    pattern_name: str
    
    # Pattern characteristics
    liquidity_pattern: Dict
    holder_pattern: Dict
    price_pattern: Dict
    volume_pattern: Dict
    social_pattern: Dict
    
    # Contract characteristics
    contract_features: Dict
    
    # Outcome data
    time_to_rug: int  # minutes from launch to rug
    damage_amount: float  # USD lost
    affected_holders: int
    
    # Pattern metadata
    confidence: float  # 0-1 how reliable this pattern is
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int


@dataclass
class RugMemoryEntry:
    """Entry in the rug memory database"""
    token_address: str
    token_symbol: str
    rug_type: RugType
    rug_date: datetime
    
    # Token characteristics at rug time
    characteristics: Dict
    
    # Rug details
    damage_usd: float
    time_active: int  # minutes from launch to rug
    warning_signs: List[str]
    
    # Pattern fingerprint
    pattern_hash: str


@dataclass
class RiskAssessment:
    """Risk assessment result for a token"""
    token_address: str
    risk_level: RiskLevel
    risk_score: float  # 0-1
    
    # Matched patterns
    matched_patterns: List[str]
    pattern_similarities: List[float]
    
    # Specific risk factors
    risk_factors: List[str]
    warning_signs: List[str]
    
    # Recommendation
    recommendation: str  # "AVOID", "CAUTION", "PROCEED"
    confidence: float


class RugMemorySystem:
    """Memory system for rug pull patterns and scam detection"""
    
    def __init__(self):
        self.logger = logging.getLogger("RugMemory")
        
        # Pattern database
        self.rug_patterns: Dict[str, RugPattern] = {}
        self.rug_memories: Dict[str, RugMemoryEntry] = {}
        
        # ML models for pattern detection
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pattern_clusterer = DBSCAN(eps=0.3, min_samples=3)
        self.scaler = StandardScaler()
        
        # Pattern matching thresholds
        self.similarity_threshold = 0.75
        self.high_risk_threshold = 0.8
        self.critical_risk_threshold = 0.9
        
        # Known rug patterns
        self._initialize_known_patterns()
        
    def _initialize_known_patterns(self):
        """Initialize with known rug pull patterns"""
        
        # Common honeypot pattern
        self.rug_patterns["honeypot_basic"] = RugPattern(
            pattern_id="honeypot_basic",
            rug_type=RugType.HONEYPOT,
            pattern_name="Basic Honeypot",
            liquidity_pattern={
                "min_liquidity": 1000,
                "max_liquidity": 50000,
                "locked_percentage": 0  # Usually unlocked
            },
            holder_pattern={
                "top_holder_min": 30,  # Top holder >30%
                "holder_count_max": 100,  # Few holders
                "concentrated_ownership": True
            },
            price_pattern={
                "initial_pump": True,
                "rapid_decline": True,
                "volatility_high": True
            },
            volume_pattern={
                "fake_volume": True,
                "bot_trading": True
            },
            social_pattern={
                "excessive_hype": True,
                "bot_comments": True,
                "fake_endorsements": True
            },
            contract_features={
                "sell_disabled": True,
                "high_sell_tax": True,
                "blacklist_function": True
            },
            time_to_rug=60,  # Usually quick
            damage_amount=10000,
            affected_holders=50,
            confidence=0.9,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1
        )
        
        # LP drain pattern
        self.rug_patterns["lp_drain_classic"] = RugPattern(
            pattern_id="lp_drain_classic",
            rug_type=RugType.LP_DRAIN,
            pattern_name="Classic LP Drain",
            liquidity_pattern={
                "initial_liquidity_high": True,
                "sudden_decrease": True,
                "drain_percentage": 90  # >90% drained
            },
            holder_pattern={
                "dev_wallet_identifiable": True,
                "large_dev_holdings": True
            },
            price_pattern={
                "sudden_crash": True,
                "price_drop_percent": 95
            },
            volume_pattern={
                "volume_spike_before_rug": True
            },
            social_pattern={
                "dev_disappearance": True,
                "social_shutdown": True
            },
            contract_features={
                "lp_not_locked": True,
                "dev_has_control": True
            },
            time_to_rug=1440,  # Usually within 24h
            damage_amount=100000,
            affected_holders=500,
            confidence=0.95,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1
        )
        
    async def analyze_token_risk(self, token_data: Dict) -> RiskAssessment:
        """Analyze token for rug pull risk based on memory patterns"""
        
        token_address = token_data.get("token_address", "")
        
        # Check if we've seen this exact token before (known rug)
        if token_address in self.rug_memories:
            return RiskAssessment(
                token_address=token_address,
                risk_level=RiskLevel.CRITICAL,
                risk_score=1.0,
                matched_patterns=["known_rug"],
                pattern_similarities=[1.0],
                risk_factors=["Previously identified as rug pull"],
                warning_signs=["KNOWN RUG - DO NOT TRADE"],
                recommendation="AVOID",
                confidence=1.0
            )
        
        # Extract features for pattern matching
        features = self._extract_token_features(token_data)
        
        # Find similar patterns
        matched_patterns, similarities = self._find_similar_patterns(features)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(features, matched_patterns, similarities)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Identify specific risk factors
        risk_factors = self._identify_risk_factors(features, matched_patterns)
        
        # Generate warning signs
        warning_signs = self._generate_warning_signs(features, matched_patterns)
        
        # Make recommendation
        recommendation = self._make_recommendation(risk_level, risk_score)
        
        assessment = RiskAssessment(
            token_address=token_address,
            risk_level=risk_level,
            risk_score=risk_score,
            matched_patterns=[p.pattern_id for p in matched_patterns],
            pattern_similarities=similarities,
            risk_factors=risk_factors,
            warning_signs=warning_signs,
            recommendation=recommendation,
            confidence=self._calculate_confidence(matched_patterns, similarities)
        )
        
        self.logger.info(
            f"Rug risk assessment for {token_data.get('token_symbol', 'UNKNOWN')}: "
            f"{risk_level.value.upper()} ({risk_score:.2f})"
        )
        
        return assessment
        
    def _extract_token_features(self, token_data: Dict) -> np.ndarray:
        """Extract numerical features for pattern matching"""
        
        features = []
        
        # Liquidity features
        features.append(token_data.get("liquidity_usd", 0))
        features.append(token_data.get("liquidity_locked_percent", 0))
        
        # Holder features
        features.append(token_data.get("holder_count", 0))
        features.append(token_data.get("top_holder_percent", 0))
        features.append(token_data.get("top_10_holder_percent", 0))
        
        # Price features
        features.append(token_data.get("price_change_5m", 0))
        features.append(token_data.get("price_change_1h", 0))
        features.append(token_data.get("price_change_24h", 0))
        features.append(token_data.get("volatility_score", 0))
        
        # Volume features
        features.append(token_data.get("volume_1h_usd", 0))
        features.append(token_data.get("volume_24h_usd", 0))
        features.append(token_data.get("volume_to_liquidity_ratio", 0))
        
        # Age and activity
        features.append(token_data.get("pool_age_seconds", 0))
        features.append(token_data.get("transaction_count_24h", 0))
        
        # Tax features
        features.append(token_data.get("buy_tax_percent", 0))
        features.append(token_data.get("sell_tax_percent", 0))
        
        # Social features
        features.append(token_data.get("social_mentions", 0))
        features.append(token_data.get("bot_activity_score", 0))
        
        return np.array(features)
        
    def _find_similar_patterns(self, features: np.ndarray) -> Tuple[List[RugPattern], List[float]]:
        """Find rug patterns similar to current token features"""
        
        similar_patterns = []
        similarities = []
        
        for pattern in self.rug_patterns.values():
            # Convert pattern to feature vector
            pattern_features = self._pattern_to_features(pattern)
            
            # Calculate similarity
            similarity = self._calculate_similarity(features, pattern_features)
            
            if similarity > self.similarity_threshold:
                similar_patterns.append(pattern)
                similarities.append(similarity)
                
        # Sort by similarity
        if similar_patterns:
            sorted_pairs = sorted(zip(similar_patterns, similarities), 
                                key=lambda x: x[1], reverse=True)
            similar_patterns, similarities = zip(*sorted_pairs)
            
        return list(similar_patterns), list(similarities)
        
    def _pattern_to_features(self, pattern: RugPattern) -> np.ndarray:
        """Convert rug pattern to feature vector"""
        
        # This is a simplified conversion - in production, would be more sophisticated
        features = []
        
        # Liquidity pattern features
        liq_pattern = pattern.liquidity_pattern
        features.append(liq_pattern.get("min_liquidity", 0))
        features.append(liq_pattern.get("locked_percentage", 0))
        
        # Holder pattern features
        holder_pattern = pattern.holder_pattern
        features.append(holder_pattern.get("holder_count_max", 1000))
        features.append(holder_pattern.get("top_holder_min", 0))
        features.append(50)  # Placeholder for top 10 holders
        
        # Price pattern features (convert boolean to numeric)
        price_pattern = pattern.price_pattern
        features.extend([
            20 if price_pattern.get("initial_pump") else 0,  # 5m change
            0,   # 1h change placeholder
            -50 if price_pattern.get("rapid_decline") else 0,  # 24h change
            0.8 if price_pattern.get("volatility_high") else 0.3  # volatility
        ])
        
        # Volume features (placeholders)
        features.extend([1000, 5000, 0.2])  # volume features
        
        # Age and activity
        features.append(pattern.time_to_rug * 60)  # Convert minutes to seconds
        features.append(100)  # Transaction count placeholder
        
        # Tax features
        contract_features = pattern.contract_features
        features.append(5 if contract_features.get("high_buy_tax") else 0)
        features.append(10 if contract_features.get("high_sell_tax") else 0)
        
        # Social features
        social_pattern = pattern.social_pattern
        features.append(1000 if social_pattern.get("excessive_hype") else 100)
        features.append(0.8 if social_pattern.get("bot_comments") else 0.2)
        
        return np.array(features)
        
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        
        # Normalize features
        if len(features1) != len(features2):
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
        # Handle division by zero
        features1 = np.nan_to_num(features1)
        features2 = np.nan_to_num(features2)
        
        # Calculate cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return max(0, similarity)  # Ensure non-negative
        
    def _calculate_risk_score(self, features: np.ndarray, patterns: List[RugPattern], 
                            similarities: List[float]) -> float:
        """Calculate overall risk score"""
        
        if not patterns:
            return 0.1  # Low base risk for unknown patterns
            
        # Weight risk by pattern confidence and similarity
        weighted_risk = 0.0
        total_weight = 0.0
        
        for pattern, similarity in zip(patterns, similarities):
            # Risk contribution from this pattern
            pattern_risk = similarity * pattern.confidence
            
            # Weight by pattern severity (time to rug)
            severity_weight = 1.0
            if pattern.time_to_rug < 60:  # Very fast rugs are more dangerous
                severity_weight = 1.5
            elif pattern.time_to_rug > 1440:  # Slow rugs less immediate danger
                severity_weight = 0.8
                
            weight = pattern_risk * severity_weight
            weighted_risk += weight
            total_weight += weight
            
        if total_weight == 0:
            return 0.1
            
        # Normalize to 0-1 range
        risk_score = min(weighted_risk / total_weight, 1.0)
        
        # Add base risk factors
        base_risk = self._calculate_base_risk(features)
        
        return min((risk_score + base_risk) / 2, 1.0)
        
    def _calculate_base_risk(self, features: np.ndarray) -> float:
        """Calculate base risk from features alone"""
        
        risk = 0.0
        
        if len(features) >= 16:  # Ensure we have enough features
            liquidity = features[0]
            top_holder = features[3]
            volatility = features[8]
            buy_tax = features[14]
            sell_tax = features[15]
            
            # Low liquidity risk
            if liquidity < 10000:
                risk += 0.3
            elif liquidity < 25000:
                risk += 0.1
                
            # High concentration risk
            if top_holder > 30:
                risk += 0.4
            elif top_holder > 20:
                risk += 0.2
                
            # High volatility risk
            if volatility > 0.8:
                risk += 0.2
                
            # High tax risk
            if sell_tax > 10:
                risk += 0.3
            if buy_tax > 10:
                risk += 0.2
                
        return min(risk, 1.0)
        
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        
        if risk_score >= self.critical_risk_threshold:
            return RiskLevel.CRITICAL
        elif risk_score >= self.high_risk_threshold:
            return RiskLevel.HIGH
        elif risk_score >= 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _identify_risk_factors(self, features: np.ndarray, patterns: List[RugPattern]) -> List[str]:
        """Identify specific risk factors"""
        
        factors = []
        
        # Pattern-based factors
        for pattern in patterns:
            if pattern.rug_type == RugType.HONEYPOT:
                factors.append("Honeypot pattern detected")
            elif pattern.rug_type == RugType.LP_DRAIN:
                factors.append("LP drain pattern detected")
            elif pattern.rug_type == RugType.DEV_DUMP:
                factors.append("Dev dump pattern detected")
                
        # Feature-based factors
        if len(features) >= 16:
            if features[0] < 10000:  # Low liquidity
                factors.append("Low liquidity")
            if features[3] > 30:  # High concentration
                factors.append("High whale concentration")
            if features[15] > 10:  # High sell tax
                factors.append("High sell tax")
                
        return factors
        
    def _generate_warning_signs(self, features: np.ndarray, patterns: List[RugPattern]) -> List[str]:
        """Generate specific warning signs"""
        
        warnings = []
        
        for pattern in patterns:
            if pattern.confidence > 0.9:
                warnings.append(f"High confidence match for {pattern.pattern_name}")
                
            if pattern.time_to_rug < 60:
                warnings.append("Pattern indicates fast rug potential")
                
        # Critical feature warnings
        if len(features) >= 16:
            if features[15] > 15:  # Very high sell tax
                warnings.append("CRITICAL: Very high sell tax detected")
            if features[3] > 50:  # Extreme concentration
                warnings.append("CRITICAL: Extreme whale concentration")
                
        return warnings
        
    def _make_recommendation(self, risk_level: RiskLevel, risk_score: float) -> str:
        """Make trading recommendation based on risk"""
        
        if risk_level == RiskLevel.CRITICAL:
            return "AVOID"
        elif risk_level == RiskLevel.HIGH:
            return "AVOID" if risk_score > 0.9 else "CAUTION"
        elif risk_level == RiskLevel.MEDIUM:
            return "CAUTION"
        else:
            return "PROCEED"
            
    def _calculate_confidence(self, patterns: List[RugPattern], similarities: List[float]) -> float:
        """Calculate confidence in the assessment"""
        
        if not patterns:
            return 0.3  # Low confidence for unknown patterns
            
        # Average of pattern confidences weighted by similarity
        total_confidence = 0.0
        total_weight = 0.0
        
        for pattern, similarity in zip(patterns, similarities):
            weight = similarity
            total_confidence += pattern.confidence * weight
            total_weight += weight
            
        return total_confidence / total_weight if total_weight > 0 else 0.3
        
    async def record_rug_pull(self, token_address: str, token_symbol: str, 
                            rug_type: RugType, characteristics: Dict):
        """Record a new rug pull for pattern learning"""
        
        # Create rug memory entry
        pattern_hash = self._create_pattern_hash(characteristics)
        
        rug_entry = RugMemoryEntry(
            token_address=token_address,
            token_symbol=token_symbol,
            rug_type=rug_type,
            rug_date=datetime.now(),
            characteristics=characteristics,
            damage_usd=characteristics.get("damage_usd", 0),
            time_active=characteristics.get("time_active", 0),
            warning_signs=characteristics.get("warning_signs", []),
            pattern_hash=pattern_hash
        )
        
        self.rug_memories[token_address] = rug_entry
        
        # Update or create pattern
        await self._update_patterns(rug_entry)
        
        self.logger.warning(f"Recorded rug pull: {token_symbol} ({rug_type.value})")
        
    def _create_pattern_hash(self, characteristics: Dict) -> str:
        """Create a hash fingerprint of the pattern"""
        
        # Extract key characteristics for hashing
        key_chars = {
            "liquidity": characteristics.get("liquidity_usd", 0),
            "top_holder": characteristics.get("top_holder_percent", 0),
            "sell_tax": characteristics.get("sell_tax_percent", 0),
            "holder_count": characteristics.get("holder_count", 0)
        }
        
        # Create hash
        pattern_str = json.dumps(key_chars, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()
        
    async def _update_patterns(self, rug_entry: RugMemoryEntry):
        """Update pattern database with new rug information"""
        
        # Find similar existing patterns
        similar_pattern = None
        for pattern in self.rug_patterns.values():
            if pattern.rug_type == rug_entry.rug_type:
                # Check if characteristics are similar
                similarity = self._calculate_characteristic_similarity(
                    rug_entry.characteristics, pattern
                )
                if similarity > 0.8:
                    similar_pattern = pattern
                    break
                    
        if similar_pattern:
            # Update existing pattern
            similar_pattern.occurrence_count += 1
            similar_pattern.last_seen = datetime.now()
            # Update confidence based on frequency
            similar_pattern.confidence = min(0.95, similar_pattern.confidence + 0.05)
        else:
            # Create new pattern
            pattern_id = f"{rug_entry.rug_type.value}_{len(self.rug_patterns)}"
            new_pattern = self._create_pattern_from_rug(pattern_id, rug_entry)
            self.rug_patterns[pattern_id] = new_pattern
            
    def _calculate_characteristic_similarity(self, chars: Dict, pattern: RugPattern) -> float:
        """Calculate similarity between characteristics and pattern"""
        
        # Simplified similarity calculation
        # In production, would use more sophisticated feature comparison
        
        similar_count = 0
        total_checks = 0
        
        # Check liquidity similarity
        if "liquidity_usd" in chars:
            total_checks += 1
            liq_min = pattern.liquidity_pattern.get("min_liquidity", 0)
            liq_max = pattern.liquidity_pattern.get("max_liquidity", float('inf'))
            if liq_min <= chars["liquidity_usd"] <= liq_max:
                similar_count += 1
                
        # Check holder similarity
        if "top_holder_percent" in chars:
            total_checks += 1
            holder_min = pattern.holder_pattern.get("top_holder_min", 0)
            if chars["top_holder_percent"] >= holder_min:
                similar_count += 1
                
        return similar_count / total_checks if total_checks > 0 else 0.0
        
    def _create_pattern_from_rug(self, pattern_id: str, rug_entry: RugMemoryEntry) -> RugPattern:
        """Create a new pattern from a rug pull entry"""
        
        chars = rug_entry.characteristics
        
        return RugPattern(
            pattern_id=pattern_id,
            rug_type=rug_entry.rug_type,
            pattern_name=f"Learned {rug_entry.rug_type.value} pattern",
            liquidity_pattern={
                "min_liquidity": chars.get("liquidity_usd", 0) * 0.8,
                "max_liquidity": chars.get("liquidity_usd", 0) * 1.2,
                "locked_percentage": chars.get("liquidity_locked_percent", 0)
            },
            holder_pattern={
                "top_holder_min": chars.get("top_holder_percent", 0) * 0.9,
                "holder_count_max": chars.get("holder_count", 0) * 1.1,
                "concentrated_ownership": chars.get("top_holder_percent", 0) > 20
            },
            price_pattern={
                "rapid_decline": True,
                "volatility_high": chars.get("volatility_score", 0) > 0.7
            },
            volume_pattern={},
            social_pattern={},
            contract_features={
                "high_sell_tax": chars.get("sell_tax_percent", 0) > 5
            },
            time_to_rug=rug_entry.time_active,
            damage_amount=rug_entry.damage_usd,
            affected_holders=chars.get("holder_count", 0),
            confidence=0.7,  # Start with moderate confidence
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1
        )
        
    def get_memory_stats(self) -> Dict:
        """Get statistics about the rug memory system"""
        
        return {
            "total_patterns": len(self.rug_patterns),
            "total_rug_memories": len(self.rug_memories),
            "pattern_types": {
                rug_type.value: len([p for p in self.rug_patterns.values() if p.rug_type == rug_type])
                for rug_type in RugType
            },
            "avg_pattern_confidence": sum(p.confidence for p in self.rug_patterns.values()) / len(self.rug_patterns) if self.rug_patterns else 0
        }


# Global instance
rug_memory_system = RugMemorySystem() 