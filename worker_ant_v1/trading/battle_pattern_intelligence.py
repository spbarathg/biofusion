"""
BATTLE PATTERN INTELLIGENCE
==========================

High-speed pattern recognition and reaction system optimized for:
- Wallet behavior analysis and learning
- Chain reaction detection
- Real-time pattern matching
- Position sizing based on pattern confidence
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging

try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.utils.real_solana_integration import ProductionSolanaClient as SolanaClient
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager as WalletManager

logger = setup_logger(__name__)

class WalletBehaviorType(Enum):
    SMART_MONEY = "smart_money"
    SNIPER_BOT = "sniper_bot"
    WHALE = "whale"
    RUG_ACCOMPLICE = "rug_accomplice"
    ORGANIC = "organic"
    UNKNOWN = "unknown"

class PatternType(Enum):
    WHALE_ENTRY = "whale_entry"
    COORDINATED_MOVE = "coordinated_move"
    SMART_MONEY_ACCUMULATION = "smart_money_accumulation"
    LIQUIDITY_DRAIN = "liquidity_drain"
    BOT_SWARM = "bot_swarm"

@dataclass
class WalletSignature:
    """Battle-focused wallet signature"""
    address: str
    
    # Core Metrics
    success_rate: float = 0.0
    profit_factor: float = 0.0
    avg_position_size: float = 0.0
    trade_frequency: float = 0.0
    reaction_speed_ms: float = 0.0
    
    # Pattern Recognition
    entry_timing_percentile: float = 0.0
    exit_timing_pattern: str = "unknown"
    gas_usage_pattern: str = "normal"
    
    # Risk Profile
    risk_score: float = 0.5
    rug_participation_count: int = 0
    suspicious_patterns_detected: int = 0
    
    # Performance
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_drawdown: float = 0.0
    
    # Intelligence
    behavior_type: WalletBehaviorType = WalletBehaviorType.UNKNOWN
    confidence_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ChainPattern:
    """Real-time chain pattern detection"""
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    urgency: int  # 1-10
    
    # Participants
    wallet_addresses: List[str]
    smart_money_count: int
    total_volume: float
    
    # Timing
    detection_time: datetime
    pattern_duration_ms: int
    
    # Analysis
    expected_move_size: float
    suggested_position_size: float
    stop_loss_level: float
    
    # Verification
    transaction_hashes: List[str]
    supporting_metrics: Dict[str, float]

class BattlePatternIntelligence:
    """High-speed pattern recognition and reaction system"""
    
    def __init__(self):
        self.logger = setup_logger("BattlePatternIntelligence")
        self.solana = SolanaClient()
        self.wallet_manager = WalletManager()
        
        # Active Monitoring
        self.active_patterns: Dict[str, ChainPattern] = {}
        self.monitored_wallets: Dict[str, WalletSignature] = {}
        self.pattern_history: deque = deque(maxlen=1000)
        
        # Performance Tracking
        self.wallet_performance: Dict[str, List[float]] = defaultdict(list)
        self.pattern_success_rate: Dict[PatternType, float] = defaultdict(float)
        
        # Pattern Detection Settings
        self.min_pattern_confidence = 0.8
        self.min_smart_money_wallets = 2
        self.max_pattern_age_ms = 500
        self.min_volume_threshold = 100  # SOL
        
        # ML Models (if available)
        self.wallet_classifier = None
        self.pattern_detector = None
        
    async def initialize(self):
        """Initialize the battle pattern system"""
        self.logger.info("🔥 Arming Battle Pattern Intelligence")
        
        if ML_AVAILABLE:
            await self._initialize_ml_models()
        else:
            self.logger.warning("ML libraries not available, using rule-based patterns only")
        
        # Start real-time monitoring
        asyncio.create_task(self._monitor_patterns())
        asyncio.create_task(self._update_wallet_intelligence())
        
        self.logger.info("⚔️ Battle Pattern System Ready")
    
    async def _initialize_ml_models(self):
        """Initialize ML models for pattern detection"""
        try:
            self.logger.info("🤖 Initializing ML models for pattern detection")
            
            # Initialize wallet behavior classifier
            self.wallet_classifier = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Initialize pattern detector (clustering)
            self.pattern_detector = DBSCAN(
                eps=0.5,
                min_samples=3
            )
            
            self.logger.info("✅ ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"ML model initialization failed: {e}")
            self.logger.warning("Falling back to rule-based pattern detection")
    
    async def analyze_wallet(self, address: str) -> WalletSignature:
        """Analyze and classify wallet behavior"""
        try:
            # Get trading history
            trades = await self.solana.get_wallet_trades(address)
            
            if not trades:
                return None
                
            # Calculate core metrics
            success_rate = len([t for t in trades if t['profit'] > 0]) / len(trades)
            profit_factor = sum([t['profit'] for t in trades if t['profit'] > 0]) / abs(sum([t['profit'] for t in trades if t['profit'] < 0]))
            avg_size = np.mean([t['size'] for t in trades])
            
            # Analyze timing patterns
            entry_speeds = [t['entry_speed_ms'] for t in trades]
            reaction_speed = np.percentile(entry_speeds, 90)  # 90th percentile speed
            
            # Create signature
            signature = WalletSignature(
                address=address,
                success_rate=success_rate,
                profit_factor=profit_factor,
                avg_position_size=avg_size,
                trade_frequency=len(trades) / ((trades[-1]['timestamp'] - trades[0]['timestamp']).total_seconds() / 3600),
                reaction_speed_ms=reaction_speed,
                entry_timing_percentile=self._calculate_entry_timing(trades),
                win_rate=success_rate,
                avg_profit_per_trade=np.mean([t['profit'] for t in trades]),
                max_drawdown=self._calculate_max_drawdown(trades)
            )
            
            # Classify behavior
            signature.behavior_type = await self._classify_wallet_behavior(signature)
            signature.confidence_score = self._calculate_confidence(trades, signature)
            
            # Update monitoring
            self.monitored_wallets[address] = signature
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Wallet analysis failed: {str(e)}")
            return None
    
    async def detect_patterns(self, token_address: str) -> List[ChainPattern]:
        """Detect active battle patterns"""
        try:
            # Get recent transactions
            txs = await self.solana.get_token_transactions(token_address)
            
            if not txs:
                return []
            
            patterns = []
            
            # Analyze transaction clusters
            clusters = self._cluster_transactions(txs)
            
            for cluster in clusters:
                # Get participating wallets
                wallets = list(set([tx['wallet'] for tx in cluster]))
                
                # Analyze wallet signatures
                signatures = [
                    self.monitored_wallets.get(w) or await self.analyze_wallet(w)
                    for w in wallets
                ]
                signatures = [s for s in signatures if s]  # Remove None
                
                # Count smart money wallets
                smart_money = len([
                    s for s in signatures 
                    if s.behavior_type == WalletBehaviorType.SMART_MONEY
                ])
                
                if smart_money >= self.min_smart_money_wallets:
                    # Calculate pattern metrics
                    volume = sum([tx['amount'] for tx in cluster])
                    duration = max([tx['timestamp'] for tx in cluster]) - min([tx['timestamp'] for tx in cluster])
                    
                    # Create pattern
                    pattern = ChainPattern(
                        pattern_id=f"{token_address}_{int(datetime.now().timestamp())}",
                        pattern_type=self._determine_pattern_type(cluster, signatures),
                        confidence=self._calculate_pattern_confidence(cluster, signatures),
                        urgency=self._calculate_urgency(cluster, signatures),
                        wallet_addresses=wallets,
                        smart_money_count=smart_money,
                        total_volume=volume,
                        detection_time=datetime.now(),
                        pattern_duration_ms=int(duration.total_seconds() * 1000),
                        expected_move_size=self._estimate_move_size(cluster, signatures),
                        suggested_position_size=self._calculate_position_size(cluster, signatures),
                        stop_loss_level=self._calculate_stop_loss(cluster),
                        transaction_hashes=[tx['hash'] for tx in cluster],
                        supporting_metrics=self._extract_supporting_metrics(cluster, signatures)
                    )
                    
                    if pattern.confidence >= self.min_pattern_confidence:
                        patterns.append(pattern)
                        self.active_patterns[pattern.pattern_id] = pattern
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {str(e)}")
            return []
    
    async def _monitor_patterns(self):
        """Real-time pattern monitoring loop"""
        while True:
            try:
                # Clean old patterns
                current_time = datetime.now()
                expired_patterns = [
                    pid for pid, p in self.active_patterns.items()
                    if (current_time - p.detection_time).total_seconds() * 1000 > self.max_pattern_age_ms
                ]
                
                for pid in expired_patterns:
                    pattern = self.active_patterns.pop(pid)
                    self.pattern_history.append(pattern)
                
                await asyncio.sleep(0.1)  # 100ms check interval
                
            except Exception as e:
                self.logger.error(f"Pattern monitoring error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _update_wallet_intelligence(self):
        """Update wallet intelligence loop"""
        while True:
            try:
                for address in list(self.monitored_wallets.keys()):
                    if (datetime.now() - self.monitored_wallets[address].last_updated).total_seconds() > 300:
                        await self.analyze_wallet(address)
                
                await asyncio.sleep(1)  # 1s update interval
                
            except Exception as e:
                self.logger.error(f"Wallet intelligence update error: {str(e)}")
                await asyncio.sleep(5)
    
    def _cluster_transactions(self, transactions: List[Dict]) -> List[List[Dict]]:
        """Cluster related transactions"""
        if not ML_AVAILABLE:
            # Fallback time-based clustering
            clusters = []
            current_cluster = []
            
            for tx in sorted(transactions, key=lambda x: x['timestamp']):
                if not current_cluster or (tx['timestamp'] - current_cluster[-1]['timestamp']).total_seconds() <= 2:
                    current_cluster.append(tx)
                else:
                    if len(current_cluster) >= 3:  # Min 3 txs per cluster
                        clusters.append(current_cluster)
                    current_cluster = [tx]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
                
            return clusters
        else:
            # Use DBSCAN for sophisticated clustering
            features = np.array([
                [
                    tx['timestamp'].timestamp(),
                    tx['amount'],
                    tx['price']
                ] for tx in transactions
            ])
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(features_scaled)
            
            clusters = []
            for label in set(clustering.labels_):
                if label != -1:  # Skip noise
                    cluster_txs = [
                        tx for i, tx in enumerate(transactions)
                        if clustering.labels_[i] == label
                    ]
                    clusters.append(cluster_txs)
            
            return clusters
    
    async def _classify_wallet_behavior(self, signature: WalletSignature) -> WalletBehaviorType:
        """Classify wallet behavior type"""
        if signature.success_rate > 0.7 and signature.profit_factor > 2.0:
            return WalletBehaviorType.SMART_MONEY
        elif signature.reaction_speed_ms < 100 and signature.trade_frequency > 20:
            return WalletBehaviorType.SNIPER_BOT
        elif signature.avg_position_size > 100:  # 100 SOL
            return WalletBehaviorType.WHALE
        elif signature.rug_participation_count > 2:
            return WalletBehaviorType.RUG_ACCOMPLICE
        elif signature.trade_frequency < 5 and signature.success_rate > 0.5:
            return WalletBehaviorType.ORGANIC
        else:
            return WalletBehaviorType.UNKNOWN
    
    def _calculate_entry_timing(self, trades: List[Dict]) -> float:
        """Calculate how early a wallet enters positions"""
        try:
            entry_times = []
            for trade in trades:
                token_launch = trade.get('token_launch_time')
                if token_launch:
                    entry_delay = (trade['timestamp'] - token_launch).total_seconds()
                    entry_times.append(entry_delay)
            
            if entry_times:
                return np.percentile(entry_times, 10)  # Use 10th percentile
            return 0.0
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        try:
            equity_curve = np.cumsum([t['profit'] for t in trades])
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            return float(np.max(drawdown))
        except:
            return 0.0
    
    def _calculate_confidence(self, trades: List[Dict], signature: WalletSignature) -> float:
        """Calculate confidence in the analysis"""
        factors = [
            len(trades) > 20,  # Enough trades
            signature.success_rate > 0.6,  # Consistent success
            signature.profit_factor > 1.5,  # Good risk/reward
            signature.max_drawdown < 0.3,  # Controlled losses
            signature.suspicious_patterns_detected < 3  # Few suspicious patterns
        ]
        return sum(factors) / len(factors)
    
    def _determine_pattern_type(self, cluster: List[Dict], signatures: List[WalletSignature]) -> PatternType:
        """Determine the type of pattern"""
        smart_money_count = len([s for s in signatures if s.behavior_type == WalletBehaviorType.SMART_MONEY])
        whale_count = len([s for s in signatures if s.behavior_type == WalletBehaviorType.WHALE])
        bot_count = len([s for s in signatures if s.behavior_type == WalletBehaviorType.SNIPER_BOT])
        
        volume = sum([tx['amount'] for tx in cluster])
        duration_ms = (max([tx['timestamp'] for tx in cluster]) - min([tx['timestamp'] for tx in cluster])).total_seconds() * 1000
        
        if smart_money_count >= 2:
            return PatternType.SMART_MONEY_ACCUMULATION
        elif whale_count >= 1 and volume > 1000:  # 1000 SOL
            return PatternType.WHALE_ENTRY
        elif bot_count >= 5 and duration_ms < 1000:  # 1 second
            return PatternType.BOT_SWARM
        elif len(set([tx['wallet'] for tx in cluster])) >= 5:
            return PatternType.COORDINATED_MOVE
        else:
            return PatternType.LIQUIDITY_DRAIN
    
    def _calculate_pattern_confidence(self, cluster: List[Dict], signatures: List[WalletSignature]) -> float:
        """Calculate confidence in pattern detection"""
        factors = [
            len(signatures) >= 3,  # Multiple participants
            any(s.behavior_type == WalletBehaviorType.SMART_MONEY for s in signatures),
            sum([tx['amount'] for tx in cluster]) > self.min_volume_threshold,
            np.mean([s.success_rate for s in signatures]) > 0.6,
            (max([tx['timestamp'] for tx in cluster]) - min([tx['timestamp'] for tx in cluster])).total_seconds() < 5
        ]
        return sum(factors) / len(factors)
    
    def _calculate_urgency(self, cluster: List[Dict], signatures: List[WalletSignature]) -> int:
        """Calculate pattern urgency (1-10)"""
        factors = [
            len([s for s in signatures if s.behavior_type == WalletBehaviorType.SMART_MONEY]) >= 2,
            sum([tx['amount'] for tx in cluster]) > self.min_volume_threshold * 2,
            any(s.reaction_speed_ms < 100 for s in signatures),
            len(set([tx['wallet'] for tx in cluster])) >= 5,
            (max([tx['timestamp'] for tx in cluster]) - min([tx['timestamp'] for tx in cluster])).total_seconds() < 2
        ]
        return min(10, sum(factors) * 2)
    
    def _estimate_move_size(self, cluster: List[Dict], signatures: List[WalletSignature]) -> float:
        """Estimate expected move size"""
        volume = sum([tx['amount'] for tx in cluster])
        smart_money_confidence = np.mean([
            s.success_rate for s in signatures 
            if s.behavior_type == WalletBehaviorType.SMART_MONEY
        ] or [0])
        
        base_expectation = 0.1  # 10% base case
        volume_factor = min(1, volume / (self.min_volume_threshold * 5))
        smart_money_factor = smart_money_confidence * 0.5
        
        return base_expectation * (1 + volume_factor + smart_money_factor)
    
    def _calculate_position_size(self, cluster: List[Dict], signatures: List[WalletSignature]) -> float:
        """Calculate suggested position size"""
        confidence = self._calculate_pattern_confidence(cluster, signatures)
        urgency = self._calculate_urgency(cluster, signatures)
        smart_money_count = len([s for s in signatures if s.behavior_type == WalletBehaviorType.SMART_MONEY])
        
        base_size = 0.01  # 1% base case
        confidence_factor = confidence * 2
        urgency_factor = urgency / 10
        smart_money_factor = min(1, smart_money_count / 3)
        
        position_size = base_size * (1 + confidence_factor + urgency_factor + smart_money_factor)
        return min(0.25, max(0.01, position_size))  # Cap between 1-25%
    
    def _calculate_stop_loss(self, cluster: List[Dict]) -> float:
        """Calculate suggested stop loss level"""
        prices = [tx['price'] for tx in cluster]
        price_range = max(prices) - min(prices)
        avg_price = np.mean(prices)
        
        return 0.05  # Fixed 5% stop loss for now
        # TODO: Implement dynamic stop loss based on volatility
    
    def _extract_supporting_metrics(self, cluster: List[Dict], signatures: List[WalletSignature]) -> Dict[str, float]:
        """Extract supporting metrics for pattern"""
        return {
            'volume': sum([tx['amount'] for tx in cluster]),
            'unique_wallets': len(set([tx['wallet'] for tx in cluster])),
            'smart_money_ratio': len([s for s in signatures if s.behavior_type == WalletBehaviorType.SMART_MONEY]) / len(signatures),
            'avg_success_rate': np.mean([s.success_rate for s in signatures]),
            'max_wallet_size': max([s.avg_position_size for s in signatures]),
            'pattern_duration_ms': (max([tx['timestamp'] for tx in cluster]) - min([tx['timestamp'] for tx in cluster])).total_seconds() * 1000
        }

async def create_battle_intelligence() -> BattlePatternIntelligence:
    """Create and initialize battle pattern intelligence"""
    intelligence = BattlePatternIntelligence()
    await intelligence.initialize()
    return intelligence 