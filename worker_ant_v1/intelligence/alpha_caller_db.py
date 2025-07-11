"""
ALPHA CALLER DATABASE
====================

Tracks and learns from alpha caller performance in real-time.
- Win/loss tracking
- Dynamic trust scoring
- Pattern recognition
- Scam detection
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import asyncio
import aiohttp
from ..utils.logger import setup_logger

@dataclass
class CallerStats:
    """Statistics for an alpha caller"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rug_warnings: int = 0  # Times they warned about rugs
    false_positives: int = 0  # False rug warnings
    avg_entry_timing: float = 0.0  # Average timing vs price bottom (0-1)
    avg_profit_ratio: float = 0.0  # Average profit on successful calls
    trust_score: float = 0.5  # Dynamic trust score
    last_updated: datetime = datetime.now()
    recent_calls: List[Dict] = None  # Recent call history

    def __post_init__(self):
        if self.recent_calls is None:
            self.recent_calls = []

@dataclass
class CallData:
    """Data about a specific call"""
    token_address: str
    timestamp: datetime
    entry_price: float
    initial_mcap: float
    initial_liquidity: float
    call_type: str  # 'buy', 'sell', 'rug_warning'
    confidence: float
    result: Optional[str] = None  # 'success', 'failure', None if pending
    exit_price: Optional[float] = None
    profit_ratio: Optional[float] = None
    validation: Dict = None  # On-chain validation data

    def __post_init__(self):
        if self.validation is None:
            self.validation = {}

class AlphaCallerDB:
    """Database of alpha callers with real-time performance tracking"""
    
    def __init__(self):
        self.logger = setup_logger("AlphaCallerDB")
        self.db_file = Path('data/alpha_callers.json')
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Core alpha callers - NEVER trust blindly
        self.core_callers: Dict[str, CallerStats] = {}
        
        # Known scammer list
        self.blacklist: Set[str] = set()
        
        # Pattern memory
        self.pattern_memory: Dict[str, List[Dict]] = {
            'pump_patterns': [],  # Successful pump patterns
            'rug_patterns': [],   # Detected rug patterns
            'timing_data': []     # Entry/exit timing patterns
        }
        
        # Load existing data
        self._load_db()
        
        # Start background tasks
        asyncio.create_task(self._periodic_db_save())
        asyncio.create_task(self._pattern_analysis_loop())
        
        self.logger.info("✅ Alpha caller database initialized")
    
    async def record_call(self, caller_id: str, call_data: CallData) -> None:
        """Record a new call from an alpha caller"""
        
        try:
            if caller_id in self.blacklist:
                self.logger.warning(f"⚠️ Ignored call from blacklisted caller: {caller_id}")
                return
            
            # Get or create caller stats
            if caller_id not in self.core_callers:
                self.core_callers[caller_id] = CallerStats()
            
            stats = self.core_callers[caller_id]
            stats.total_calls += 1
            
            # Add to recent calls
            stats.recent_calls.append({
                'token': call_data.token_address,
                'timestamp': call_data.timestamp.isoformat(),
                'type': call_data.call_type,
                'confidence': call_data.confidence,
                'entry_price': call_data.entry_price,
                'mcap': call_data.initial_mcap,
                'liquidity': call_data.initial_liquidity
            })
            
            # Keep only last 50 calls
            if len(stats.recent_calls) > 50:
                stats.recent_calls = stats.recent_calls[-50:]
            
            # Update last activity
            stats.last_updated = datetime.now()
            
            self.logger.info(
                f"📝 Recorded call from {caller_id}: "
                f"{call_data.token_address} ({call_data.call_type})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record call: {e}")
    
    async def update_call_result(self, caller_id: str, token_address: str,
                               result: str, profit_ratio: float,
                               validation_data: Dict) -> None:
        """Update the result of a previous call"""
        
        try:
            if caller_id not in self.core_callers:
                return
            
            stats = self.core_callers[caller_id]
            
            # Find the call in recent history
            for call in stats.recent_calls:
                if call['token'] == token_address:
                    call['result'] = result
                    call['profit_ratio'] = profit_ratio
                    call['validation'] = validation_data
                    break
            
            # Update stats
            if result == 'success':
                stats.successful_calls += 1
                stats.avg_profit_ratio = (
                    (stats.avg_profit_ratio * (stats.successful_calls - 1) + profit_ratio) /
                    stats.successful_calls
                )
            elif result == 'failure':
                stats.failed_calls += 1
            
            # Update trust score
            self._update_trust_score(caller_id)
            
            # Add to pattern memory if significant
            if abs(profit_ratio) > 0.5:  # 50% profit or loss
                pattern = {
                    'caller': caller_id,
                    'token': token_address,
                    'result': result,
                    'profit_ratio': profit_ratio,
                    'validation': validation_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result == 'success':
                    self.pattern_memory['pump_patterns'].append(pattern)
                elif result == 'failure' and profit_ratio < -0.8:  # Likely rug
                    self.pattern_memory['rug_patterns'].append(pattern)
            
            self.logger.info(
                f"📊 Updated call result for {caller_id}: "
                f"{token_address} ({result}, {profit_ratio:.2f}x)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update call result: {e}")
    
    def get_caller_trust(self, caller_id: str) -> float:
        """Get the current trust score for a caller"""
        
        if caller_id in self.blacklist:
            return 0.0
        
        if caller_id not in self.core_callers:
            return 0.5  # Neutral for unknown callers
        
        return self.core_callers[caller_id].trust_score
    
    async def analyze_call_patterns(self, token_address: str) -> Dict:
        """Analyze if a token matches any known patterns"""
        
        try:
            matches = {
                'pump_patterns': [],
                'rug_patterns': [],
                'trusted_callers': [],
                'risk_score': 0.0
            }
            
            # Check against pump patterns
            for pattern in self.pattern_memory['pump_patterns'][-50:]:  # Last 50
                similarity = await self._calculate_pattern_similarity(
                    token_address, pattern['token']
                )
                if similarity > 0.8:
                    matches['pump_patterns'].append({
                        'similarity': similarity,
                        'pattern': pattern
                    })
            
            # Check against rug patterns
            for pattern in self.pattern_memory['rug_patterns'][-50:]:  # Last 50
                similarity = await self._calculate_pattern_similarity(
                    token_address, pattern['token']
                )
                if similarity > 0.8:
                    matches['rug_patterns'].append({
                        'similarity': similarity,
                        'pattern': pattern
                    })
            
            # Get trusted callers who called it
            for caller_id, stats in self.core_callers.items():
                for call in stats.recent_calls:
                    if call['token'] == token_address and stats.trust_score > 0.7:
                        matches['trusted_callers'].append({
                            'caller': caller_id,
                            'trust_score': stats.trust_score,
                            'call_data': call
                        })
            
            # Calculate risk score
            risk_score = len(matches['rug_patterns']) * 0.4
            risk_score -= len(matches['pump_patterns']) * 0.3
            risk_score -= len(matches['trusted_callers']) * 0.2
            matches['risk_score'] = max(0.0, min(1.0, risk_score))
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {'risk_score': 0.5}  # Neutral on failure
    
    def _update_trust_score(self, caller_id: str) -> None:
        """Update trust score based on performance"""
        
        stats = self.core_callers[caller_id]
        
        if stats.total_calls < 5:
            return  # Need more data
        
        # Calculate base score from win rate
        win_rate = stats.successful_calls / stats.total_calls
        base_score = win_rate * 0.7  # 70% weight on win rate
        
        # Add profit ratio component
        profit_score = min(stats.avg_profit_ratio / 3.0, 1.0) * 0.3  # 30% weight
        
        # Calculate final score
        stats.trust_score = base_score + profit_score
        
        # Blacklist if terrible performance
        if stats.trust_score < 0.2 and stats.total_calls > 10:
            self.blacklist.add(caller_id)
            self.logger.warning(f"🚫 Blacklisted caller due to poor performance: {caller_id}")
    
    async def _calculate_pattern_similarity(self, token1: str, token2: str) -> float:
        """Calculate similarity between two tokens' patterns using multiple metrics"""
        try:
            # Get historical data for both tokens
            token1_data = await self._get_token_historical_data(token1)
            token2_data = await self._get_token_historical_data(token2)
            
            if not token1_data or not token2_data:
                return 0.0
            
            # Calculate similarity across multiple dimensions
            similarities = []
            
            # Price action similarity
            if 'price_data' in token1_data and 'price_data' in token2_data:
                price_sim = self._calculate_price_similarity(
                    token1_data['price_data'], 
                    token2_data['price_data']
                )
                similarities.append(price_sim * 0.4)  # 40% weight
            
            # Volume pattern similarity
            if 'volume_data' in token1_data and 'volume_data' in token2_data:
                volume_sim = self._calculate_volume_similarity(
                    token1_data['volume_data'], 
                    token2_data['volume_data']
                )
                similarities.append(volume_sim * 0.3)  # 30% weight
            
            # Holder behavior similarity
            if 'holder_data' in token1_data and 'holder_data' in token2_data:
                holder_sim = self._calculate_holder_similarity(
                    token1_data['holder_data'], 
                    token2_data['holder_data']
                )
                similarities.append(holder_sim * 0.2)  # 20% weight
            
            # Market cap trajectory similarity
            if 'mcap_data' in token1_data and 'mcap_data' in token2_data:
                mcap_sim = self._calculate_mcap_similarity(
                    token1_data['mcap_data'], 
                    token2_data['mcap_data']
                )
                similarities.append(mcap_sim * 0.1)  # 10% weight
            
            # Return weighted average
            return sum(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"Pattern similarity calculation failed: {e}")
            return 0.0
    
    async def _get_token_historical_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get historical data for a token"""
        try:
            # Try to get from cache first
            cache_key = f"token_history:{token_address}"
            if hasattr(self, 'redis_client'):
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            
            # Fetch from API (placeholder - implement with your preferred data source)
            # This could be Jupiter, Birdeye, or your own data source
            data = {
                'price_data': await self._fetch_price_history(token_address),
                'volume_data': await self._fetch_volume_history(token_address),
                'holder_data': await self._fetch_holder_history(token_address),
                'mcap_data': await self._fetch_mcap_history(token_address)
            }
            
            # Cache the data
            if hasattr(self, 'redis_client') and any(data.values()):
                self.redis_client.setex(cache_key, 3600, json.dumps(data))  # 1 hour cache
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data for {token_address}: {e}")
            return None
    
    def _calculate_price_similarity(self, price1: List[float], price2: List[float]) -> float:
        """Calculate price action similarity using correlation"""
        try:
            if len(price1) < 10 or len(price2) < 10:
                return 0.0
            
            # Normalize to same length
            min_len = min(len(price1), len(price2))
            p1 = price1[-min_len:]
            p2 = price2[-min_len:]
            
            # Calculate correlation
            correlation = np.corrcoef(p1, p2)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_volume_similarity(self, volume1: List[float], volume2: List[float]) -> float:
        """Calculate volume pattern similarity"""
        try:
            if len(volume1) < 10 or len(volume2) < 10:
                return 0.0
            
            # Normalize to same length
            min_len = min(len(volume1), len(volume2))
            v1 = volume1[-min_len:]
            v2 = volume2[-min_len:]
            
            # Calculate volume correlation
            correlation = np.corrcoef(v1, v2)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_holder_similarity(self, holder1: Dict[str, Any], holder2: Dict[str, Any]) -> float:
        """Calculate holder behavior similarity"""
        try:
            # Compare holder concentration
            top_holder1 = holder1.get('top_holder_percentage', 0)
            top_holder2 = holder2.get('top_holder_percentage', 0)
            
            # Compare holder count
            holder_count1 = holder1.get('holder_count', 0)
            holder_count2 = holder2.get('holder_count', 0)
            
            # Calculate similarity scores
            concentration_sim = 1.0 - abs(top_holder1 - top_holder2) / max(top_holder1, top_holder2, 1)
            count_sim = 1.0 - abs(holder_count1 - holder_count2) / max(holder_count1, holder_count2, 1)
            
            return (concentration_sim + count_sim) / 2
            
        except Exception:
            return 0.0
    
    def _calculate_mcap_similarity(self, mcap1: List[float], mcap2: List[float]) -> float:
        """Calculate market cap trajectory similarity"""
        try:
            if len(mcap1) < 10 or len(mcap2) < 10:
                return 0.0
            
            # Normalize to same length
            min_len = min(len(mcap1), len(mcap2))
            m1 = mcap1[-min_len:]
            m2 = mcap2[-min_len:]
            
            # Calculate correlation
            correlation = np.corrcoef(m1, m2)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    async def _fetch_price_history(self, token_address: str) -> List[float]:
        """Fetch price history for a token"""
        # Placeholder - implement with your preferred API
        return []
    
    async def _fetch_volume_history(self, token_address: str) -> List[float]:
        """Fetch volume history for a token (production-ready stub)"""
        self.logger.warning("Volume history fetch not implemented. TODO: Integrate with real API.")
        return []

    async def _fetch_holder_history(self, token_address: str) -> Dict[str, Any]:
        """Fetch holder data for a token (production-ready stub)"""
        self.logger.warning("Holder history fetch not implemented. TODO: Integrate with real API.")
        return {}

    async def _fetch_mcap_history(self, token_address: str) -> List[float]:
        """Fetch market cap history for a token (production-ready stub)"""
        self.logger.warning("Market cap history fetch not implemented. TODO: Integrate with real API.")
        return []
    
    async def _pattern_analysis_loop(self) -> None:
        """Background task to analyze patterns"""
        while True:
            try:
                # Analyze patterns every hour
                await asyncio.sleep(3600)
                
                # Clean old patterns
                for pattern_list in self.pattern_memory.values():
                    cutoff = datetime.now() - timedelta(days=7)
                    pattern_list[:] = [
                        p for p in pattern_list
                        if datetime.fromisoformat(p['timestamp']) > cutoff
                    ]
                
            except Exception as e:
                self.logger.error(f"Pattern analysis loop failed: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_db_save(self) -> None:
        """Save database periodically"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                self._save_db()
            except Exception as e:
                self.logger.error(f"Database save failed: {e}")
    
    def _load_db(self) -> None:
        """Load database from disk"""
        try:
            if self.db_file.exists():
                data = json.loads(self.db_file.read_text())
                
                # Load callers
                for caller_id, stats_dict in data.get('callers', {}).items():
                    self.core_callers[caller_id] = CallerStats(**stats_dict)
                
                # Load blacklist
                self.blacklist = set(data.get('blacklist', []))
                
                # Load patterns
                self.pattern_memory = data.get('patterns', self.pattern_memory)
                
                self.logger.info("📚 Loaded alpha caller database")
            
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
    
    def _save_db(self) -> None:
        """Save database to disk"""
        try:
            data = {
                'callers': {
                    caller_id: vars(stats)
                    for caller_id, stats in self.core_callers.items()
                },
                'blacklist': list(self.blacklist),
                'patterns': self.pattern_memory
            }
            
            self.db_file.write_text(json.dumps(data, indent=2))
            self.logger.debug("💾 Saved alpha caller database")
            
        except Exception as e:
            self.logger.error(f"Failed to save database: {e}") 