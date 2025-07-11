"""
UNIFIED WALLET MANAGEMENT SYSTEM
================================

Consolidates all wallet management functionality into a single, coherent system.
Implements 10 independent evolving mini-wallets with genetic traits and nightly evolution.
"""

import asyncio
import secrets
import base58
import json
import gc
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solana.transaction import Transaction

from worker_ant_v1.core.unified_config import get_security_config, get_network_config, get_trading_config, mask_sensitive_value
from worker_ant_v1.utils.logger import setup_logger

class WalletStatus(Enum):
    ACTIVE = "active"
    ROTATING = "rotating"
    RETIRED = "retired"
    EMERGENCY = "emergency"
    EVOLVING = "evolving"

class WalletTier(Enum):
    STARTER = "starter"        # New wallets with basic capabilities
    INTERMEDIATE = "intermediate"  # Proven wallets with good performance
    ADVANCED = "advanced"      # High-performing wallets
    MASTER = "master"          # Top-tier wallets with maximum privileges
    LEGENDARY = "legendary"    # Elite evolved wallets

@dataclass
class EvolutionGenetics:
    """Genetic traits for wallet evolution (from archived evolution_engine.py)"""
    
    # Trading traits
    aggression: float = 0.5        # 0.0 (conservative) to 1.0 (aggressive)
    risk_tolerance: float = 0.5    # Risk appetite
    patience: float = 0.5          # Hold duration preference
    signal_trust: float = 0.5      # Trust in different signal types
    
    # Learning traits  
    adaptation_rate: float = 0.5   # How quickly wallet learns
    memory_strength: float = 0.5   # How well wallet remembers patterns
    pattern_recognition: float = 0.5 # Ability to spot patterns
    
    # Social traits
    herd_immunity: float = 0.5     # Resistance to crowd psychology
    leadership: float = 0.5        # Influence on other wallets
    
    # New enhanced traits for improved evolution
    delayed_reward_sensitivity: float = 0.5  # Ability to consider long-term profits
    experiment_memory: float = 0.5           # Memory of failed experiments
    variance_tolerance: float = 0.5          # Tolerance for performance variance
    consensus_weight: float = 0.5            # Influence in wallet voting
    
    # Mutation rate for offspring
    mutation_rate: float = 0.1
    
    def mutate(self) -> 'EvolutionGenetics':
        """Create mutated version for evolution with enhanced traits"""
        mutated = EvolutionGenetics()
        
        traits = ['aggression', 'risk_tolerance', 'patience', 'signal_trust',
                 'adaptation_rate', 'memory_strength', 'pattern_recognition',
                 'herd_immunity', 'leadership', 'delayed_reward_sensitivity',
                 'experiment_memory', 'variance_tolerance', 'consensus_weight']
        
        for trait in traits:
            current_value = getattr(self, trait)
            if random.random() < self.mutation_rate:
                # Mutate with normal distribution around current value
                mutation = np.random.normal(0, 0.2)
                new_value = np.clip(current_value + mutation, 0.0, 1.0)
                setattr(mutated, trait, new_value)
            else:
                setattr(mutated, trait, current_value)
        
        return mutated

@dataclass
class DelayedRewardTracker:
    """Track trades that become profitable over time"""
    trade_id: str
    wallet_id: str
    entry_time: datetime
    entry_price: float
    current_price: float
    initial_loss: float = 0.0
    max_loss: float = 0.0
    time_to_profit: Optional[timedelta] = None
    final_profit: float = 0.0
    is_profitable: bool = False
    has_recovered: bool = False

@dataclass  
class ExperimentMemory:
    """Memory of failed experiments to avoid repetition"""
    experiment_id: str
    strategy_type: str
    market_conditions: Dict[str, Any]
    failure_reason: str
    failure_cost: float
    timestamp: datetime
    confidence_loss: float = 0.0

@dataclass
class WalletConsensusData:
    """Data for wallet voting and consensus system"""
    current_signals: Dict[str, float] = field(default_factory=dict)
    vote_history: List[Tuple[str, float, datetime]] = field(default_factory=list)  # (signal, vote, time)
    influence_score: float = 0.5
    consensus_accuracy: float = 0.5

@dataclass
class SignalTrust:
    """Dynamic trust system for different signal types (from archived evolution_engine.py)"""
    
    # Signal source trust levels (0.0 to 1.0)
    technical_analysis: float = 0.6
    social_sentiment: float = 0.4
    volume_analysis: float = 0.7
    liquidity_analysis: float = 0.8
    ml_predictions: float = 0.5
    caller_reputation: float = 0.3
    
    # Trust decay and reinforcement
    trust_decay_rate: float = 0.01   # Daily decay if no reinforcement
    success_boost: float = 0.1       # Boost on successful trade
    failure_penalty: float = 0.15    # Penalty on failed trade
    
    def update_trust(self, signal_type: str, success: bool, impact: float = 1.0):
        """Update trust based on trade outcome"""
        if not hasattr(self, signal_type):
            return
            
        current_trust = getattr(self, signal_type)
        
        if success:
            # Boost trust on success
            new_trust = min(1.0, current_trust + (self.success_boost * impact))
        else:
            # Penalize trust on failure
            new_trust = max(0.0, current_trust - (self.failure_penalty * impact))
        
        setattr(self, signal_type, new_trust)
    
    def daily_decay(self):
        """Apply daily trust decay"""
        signals = ['technical_analysis', 'social_sentiment', 'volume_analysis',
                  'liquidity_analysis', 'ml_predictions', 'caller_reputation']
        
        for signal in signals:
            current = getattr(self, signal)
            new_value = max(0.1, current * (1 - self.trust_decay_rate))
            setattr(self, signal, new_value)

@dataclass
class WalletCredentials:
    """Secure wallet credentials with encrypted private key"""
    public_key: str
    encrypted_private_key: str
    salt: str
    created_at: datetime
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class WalletPerformance:
    """Wallet performance tracking with evolution metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    total_profit_sol: float = 0.0
    total_volume_sol: float = 0.0
    win_rate: float = 0.0
    average_profit_per_trade: float = 0.0
    risk_score: float = 0.0
    fitness_score: float = 0.0  # Evolution fitness
    evolution_generation: int = 1  # Generation number
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Enhanced performance tracking
    delayed_rewards: List[DelayedRewardTracker] = field(default_factory=list)
    experiment_failures: List[ExperimentMemory] = field(default_factory=list)
    variance_metrics: Dict[str, float] = field(default_factory=dict)
    long_term_performance: Dict[str, float] = field(default_factory=dict)
    survival_threshold: float = 0.3  # Dynamic threshold based on volatility

@dataclass
class Wallet:
    """Unified wallet representation with evolution capabilities"""
    id: str
    public_key: str
    tier: WalletTier
    status: WalletStatus
    balance_sol: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    performance: WalletPerformance = field(default_factory=WalletPerformance)
    genetics: EvolutionGenetics = field(default_factory=EvolutionGenetics)
    signal_trust: SignalTrust = field(default_factory=SignalTrust)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced wallet features
    consensus_data: WalletConsensusData = field(default_factory=WalletConsensusData)
    learning_phase: bool = False  # Whether in learning phase (different thresholds)
    
    def calculate_delayed_reward_score(self) -> float:
        """Calculate score based on delayed rewards that became profitable"""
        if not self.performance.delayed_rewards:
            return 0.0
            
        profitable_delayed = [dr for dr in self.performance.delayed_rewards if dr.is_profitable]
        if not profitable_delayed:
            return 0.0
            
        # Score based on recovery ability and time to profit
        total_score = 0.0
        for reward in profitable_delayed:
            time_factor = 1.0 / max(1.0, reward.time_to_profit.total_seconds() / 3600)  # Faster recovery = better
            profit_factor = reward.final_profit / max(0.01, abs(reward.initial_loss))  # Recovery ratio
            total_score += time_factor * profit_factor * self.genetics.delayed_reward_sensitivity
            
        return min(1.0, total_score / len(profitable_delayed))
    
    def has_failed_experiment(self, strategy_type: str, market_conditions: Dict[str, Any]) -> bool:
        """Check if similar experiment has failed before"""
        for memory in self.performance.experiment_failures:
            if memory.strategy_type == strategy_type:
                # Simple condition matching - could be more sophisticated
                condition_similarity = 0.0
                for key, value in market_conditions.items():
                    if key in memory.market_conditions:
                        if isinstance(value, (int, float)):
                            condition_similarity += 1.0 - abs(value - memory.market_conditions[key]) / max(value, memory.market_conditions[key], 1.0)
                        else:
                            condition_similarity += 1.0 if value == memory.market_conditions[key] else 0.0
                
                similarity_ratio = condition_similarity / max(len(market_conditions), 1)
                memory_strength = self.genetics.experiment_memory
                
                if similarity_ratio > 0.7 * memory_strength:  # Similar conditions with strong memory
                    return True
        return False

class AsyncLockManager:
    """Centralized async lock management for wallet operations"""
    
    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
    
    async def get_wallet_lock(self, wallet_id: str) -> asyncio.Lock:
        """Get or create lock for specific wallet"""
        async with self._global_lock:
            if wallet_id not in self._locks:
                self._locks[wallet_id] = asyncio.Lock()
            return self._locks[wallet_id]

class UnifiedWalletManager:
    """Enhanced wallet manager with concurrency safety"""
    
    def __init__(self):
        self.logger = setup_logger("UnifiedWalletManager")
        
        # Core systems
        self.client: Optional[AsyncClient] = None
        self.wallets: Dict[str, Wallet] = {}
        
        # Configuration
        self.trading_config = get_trading_config()
        self.security_config = get_security_config()
        
        # Evolution tracking
        self.last_evolution_cycle = datetime.now()
        self.evolution_cycle_hours = 24  # Daily evolution
        
        # Encryption (Fernet key for private key storage)
        self.encryption_key: Optional[bytes] = None
        
        # Background tasks
        self.rotation_task: Optional[asyncio.Task] = None
        self.evolution_task: Optional[asyncio.Task] = None
        
        # === CONCURRENCY SAFETY ===
        self.state_lock = asyncio.Lock()  # Global state protection
        self.balance_locks: Dict[str, asyncio.Lock] = {}  # Per-wallet balance locks
        self.performance_locks: Dict[str, asyncio.Lock] = {}  # Per-wallet performance locks
        
    async def _get_wallet_lock(self, wallet_id: str, lock_type: str = "general") -> asyncio.Lock:
        """Get wallet-specific lock for different operations"""
        if lock_type == "balance":
            if wallet_id not in self.balance_locks:
                self.balance_locks[wallet_id] = asyncio.Lock()
            return self.balance_locks[wallet_id]
        elif lock_type == "performance":
            if wallet_id not in self.performance_locks:
                self.performance_locks[wallet_id] = asyncio.Lock()
            return self.performance_locks[wallet_id]
        else:
            return await _lock_manager.get_wallet_lock(wallet_id)
    
    async def initialize(self) -> bool:
        """Initialize the wallet management system"""
        try:
            self.logger.info("🔗 Initializing Unified Wallet Manager with Evolution...")
            
            # Initialize Solana client
            await self._initialize_solana_client()
            
            # Initialize encryption
            await self._initialize_encryption()
            
            # Load existing wallets
            await self._load_existing_wallets()
            
            # Create initial wallets if none exist (EXACTLY 10 as specified)
            if not self.wallets:
                await self._create_initial_ten_wallets()
            
            # Start evolution scheduler
            if self.evolution_enabled:
                asyncio.create_task(self._evolution_scheduler())
            
            # Start rotation scheduler
            if self.security_config.wallet_rotation_enabled:
                asyncio.create_task(self._rotation_scheduler())
            
            self.logger.info(f"✅ Wallet Manager initialized with {len(self.wallets)} evolving wallets")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize wallet manager: {e}")
            return False
    
    async def _initialize_solana_client(self):
        """Initialize Solana RPC client"""
        self.client = AsyncClient(
            endpoint=self.network_config.rpc_url,
            timeout=self.network_config.rpc_timeout_seconds
        )
        
        # Test connection
        try:
            await self.client.get_version()
            self.logger.info("✅ Solana client connected")
        except Exception as e:
            self.logger.warning(f"⚠️ Solana client connection issue: {e}")
            # Try backup RPC
            if self.network_config.backup_rpc_urls:
                for backup_url in self.network_config.backup_rpc_urls:
                    try:
                        self.client = AsyncClient(
                            endpoint=backup_url,
                            timeout=self.network_config.rpc_timeout_seconds
                        )
                        await self.client.get_version()
                        self.logger.info(f"✅ Connected to backup RPC: {backup_url}")
                        break
                    except Exception:
                        continue
    
    async def _initialize_encryption(self):
        """Initialize encryption for private keys"""
        # In production, this would come from secure key management
        self.encryption_key = secrets.token_bytes(32)
        self.logger.info("🔐 Encryption initialized")
    
    async def _load_existing_wallets(self):
        """Load existing wallets from storage"""
        # In production, this would load from secure database
        # For now, we'll start fresh
        self.logger.info("📁 Loading existing wallets...")
    
    async def _create_initial_ten_wallets(self):
        """Create initial set of EXACTLY 10 wallets as specified"""
        self.logger.info("🎯 Creating initial set of 10 independent evolving mini-wallets...")
        
        # Distribution: 4 starter, 3 intermediate, 2 advanced, 1 master
        wallet_configs = [
            (WalletTier.STARTER, 4),      # 4 starter wallets
            (WalletTier.INTERMEDIATE, 3), # 3 intermediate wallets
            (WalletTier.ADVANCED, 2),     # 2 advanced wallets
            (WalletTier.MASTER, 1),       # 1 master wallet
        ]
        
        for tier, count in wallet_configs:
            for i in range(count):
                wallet_id = f"{tier.value}_{i+1}_{secrets.token_hex(4)}"
                
                # Create unique genetics for each wallet
                genetics = EvolutionGenetics()
                
                # Tier-based genetic tendencies
                if tier == WalletTier.STARTER:
                    genetics.risk_tolerance = random.uniform(0.2, 0.6)
                    genetics.aggression = random.uniform(0.1, 0.4)
                elif tier == WalletTier.INTERMEDIATE:
                    genetics.risk_tolerance = random.uniform(0.4, 0.7)
                    genetics.aggression = random.uniform(0.3, 0.6)
                elif tier == WalletTier.ADVANCED:
                    genetics.risk_tolerance = random.uniform(0.6, 0.8)
                    genetics.aggression = random.uniform(0.5, 0.8)
                elif tier == WalletTier.MASTER:
                    genetics.risk_tolerance = random.uniform(0.7, 0.9)
                    genetics.aggression = random.uniform(0.6, 0.9)
                    genetics.leadership = random.uniform(0.7, 1.0)
                
                await self.create_wallet(wallet_id, tier, genetics)
    
    async def create_wallet(self, wallet_id: str, tier: WalletTier = WalletTier.STARTER, 
                          genetics: Optional[EvolutionGenetics] = None) -> Optional[Wallet]:
        """Create a new wallet with evolution capabilities"""
        try:
            self.logger.info(f"🆕 Creating evolving wallet: {wallet_id}")
            
            # Generate keypair
            keypair = Keypair()
            public_key = str(keypair.public_key)
            private_key = base58.b58encode(keypair.secret_key).decode()
            
            # Encrypt private key
            encrypted_private_key, salt = await self._encrypt_private_key(private_key)
            
            # Create credentials
            credentials = WalletCredentials(
                public_key=public_key,
                encrypted_private_key=encrypted_private_key,
                salt=salt,
                created_at=datetime.now()
            )
            
            # Create wallet with genetics
            wallet = Wallet(
                id=wallet_id,
                public_key=public_key,
                tier=tier,
                status=WalletStatus.ACTIVE,
                genetics=genetics or EvolutionGenetics()
            )
            
            # Store wallet and credentials
            self.wallets[wallet_id] = wallet
            self.credentials[wallet_id] = credentials
            
            self.logger.info(f"✅ Created evolving wallet {wallet_id} (Tier: {tier.value})")
            return wallet
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create wallet {wallet_id}: {e}")
            return None
    
    async def _encrypt_private_key(self, private_key: str) -> Tuple[str, str]:
        """Encrypt private key for secure storage"""
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64
        
        # Generate salt
        salt = secrets.token_bytes(16)
        
        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key))
        
        # Encrypt
        fernet = Fernet(key)
        encrypted = fernet.encrypt(private_key.encode())
        
        return base64.b64encode(encrypted).decode(), base64.b64encode(salt).decode()
    
    async def get_wallet_for_trading(self, preferred_tier: Optional[WalletTier] = None) -> Optional[Wallet]:
        """Get the best available wallet for trading"""
        try:
            # Filter active wallets
            active_wallets = [w for w in self.wallets.values() if w.status == WalletStatus.ACTIVE]
            
            if not active_wallets:
                self.logger.warning("⚠️ No active wallets available")
                return None
            
            # Filter by tier if specified
            if preferred_tier:
                tier_wallets = [w for w in active_wallets if w.tier == preferred_tier]
                if tier_wallets:
                    active_wallets = tier_wallets
            
            # Select best wallet based on performance and balance
            best_wallet = max(active_wallets, key=lambda w: (
                w.performance.win_rate,
                w.balance_sol,
                w.tier.value
            ))
            
            # Update last active
            best_wallet.last_active = datetime.now()
            
            return best_wallet
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get trading wallet: {e}")
            return None
    
    async def update_wallet_balance(self, wallet_id: str) -> bool:
        """Update wallet balance from blockchain - THREAD SAFE"""
        balance_lock = await self._get_wallet_lock(wallet_id, "balance")
        
        async with balance_lock:
            try:
                if wallet_id not in self.wallets:
                    return False
                
                wallet = self.wallets[wallet_id]
                
                # Get balance from Solana
                response = await self.client.get_balance(PublicKey(wallet.public_key))
                if response.value is not None:
                    old_balance = wallet.balance_sol
                    new_balance = response.value / 1e9  # Convert lamports to SOL
                    
                    # Atomic balance update
                    wallet.balance_sol = new_balance
                    wallet.last_active = datetime.now()
                    
                    # Log significant balance changes
                    if abs(new_balance - old_balance) > 0.01:  # 0.01 SOL threshold
                        self.logger.info(f"💰 Balance updated for {wallet_id}: "
                                       f"{old_balance:.4f} → {new_balance:.4f} SOL "
                                       f"({new_balance - old_balance:+.4f})")
                    
                    return True
                
                return False
                
            except Exception as e:
                self.logger.error(f"❌ Failed to update balance for {wallet_id}: {e}")
                return False
    
    async def record_trade_result(self, wallet_id: str, profit_sol: float, volume_sol: float, success: bool,
                                signal_types_used: List[str] = None, trade_metadata: Dict[str, Any] = None):
        """Record trade result and update wallet evolution metrics with enhanced delayed reward tracking - THREAD SAFE"""
        performance_lock = await self._get_wallet_lock(wallet_id, "performance")
        
        async with performance_lock:
            try:
                if wallet_id not in self.wallets:
                    self.logger.warning(f"⚠️ Attempted to record trade for unknown wallet: {wallet_id}")
                    return
                
                wallet = self.wallets[wallet_id]
                performance = wallet.performance
                trade_metadata = trade_metadata or {}
                
                # Atomic performance update
                old_trades = performance.total_trades
                old_profit = performance.total_profit_sol
                
                performance.total_trades += 1
                performance.total_volume_sol += volume_sol
                performance.total_profit_sol += profit_sol
                
                if success:
                    performance.successful_trades += 1
                
                # NEW: Track delayed rewards for initially losing trades
                if not success and profit_sol < 0:
                    # Create delayed reward tracker for this losing trade
                    delayed_tracker = DelayedRewardTracker(
                        trade_id=trade_metadata.get('trade_id', f"trade_{performance.total_trades}"),
                        wallet_id=wallet_id,
                        entry_time=trade_metadata.get('entry_time', datetime.now()),
                        entry_price=trade_metadata.get('entry_price', 0.0),
                        current_price=trade_metadata.get('current_price', 0.0),
                        initial_loss=abs(profit_sol),
                        max_loss=abs(profit_sol)
                    )
                    performance.delayed_rewards.append(delayed_tracker)
                    
                    # Keep only recent delayed rewards (last 100)
                    if len(performance.delayed_rewards) > 100:
                        performance.delayed_rewards = performance.delayed_rewards[-100:]
                
                # Update variance metrics for long-term performance tracking
                await self._update_variance_metrics(wallet, profit_sol)
                
                # Recalculate derived metrics atomically
                performance.win_rate = performance.successful_trades / performance.total_trades
                performance.average_profit_per_trade = performance.total_profit_sol / performance.total_trades
                performance.last_updated = datetime.now()
                
                # Update signal trust based on outcome
                if signal_types_used:
                    for signal_type in signal_types_used:
                        wallet.signal_trust.update_trust(signal_type, success, abs(profit_sol))
                
                # Update wallet tier based on performance
                await self._update_wallet_tier_safe(wallet_id)
                
                self.logger.info(f"📊 Trade recorded for {wallet_id}: "
                              f"{'✅' if success else '❌'} {profit_sol:+.4f} SOL "
                              f"(Total: {old_trades + 1} trades, {old_profit + profit_sol:+.4f} SOL)")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to record trade result for {wallet_id}: {e}")
    
    async def update_delayed_reward_status(self, wallet_id: str, trade_id: str, current_price: float, current_profit: float):
        """Update status of delayed reward tracker when position becomes profitable"""
        if wallet_id not in self.wallets:
            return
            
        wallet = self.wallets[wallet_id]
        
        for delayed_reward in wallet.performance.delayed_rewards:
            if delayed_reward.trade_id == trade_id:
                delayed_reward.current_price = current_price
                
                if current_profit > 0 and not delayed_reward.is_profitable:
                    # Trade became profitable!
                    delayed_reward.is_profitable = True
                    delayed_reward.has_recovered = True
                    delayed_reward.final_profit = current_profit
                    delayed_reward.time_to_profit = datetime.now() - delayed_reward.entry_time
                    
                    self.logger.info(f"🎯 Delayed reward achieved for wallet {wallet_id}: {trade_id} recovered in {delayed_reward.time_to_profit}")
                
                # Update max loss tracking
                if current_profit < 0:
                    delayed_reward.max_loss = max(delayed_reward.max_loss, abs(current_profit))
                
                break
    
    async def _update_variance_metrics(self, wallet: Wallet, latest_profit: float):
        """Update variance metrics for long-term performance analysis"""
        # Track rolling window of profits for variance calculation
        if 'recent_profits' not in wallet.performance.variance_metrics:
            wallet.performance.variance_metrics['recent_profits'] = []
        
        recent_profits = wallet.performance.variance_metrics['recent_profits']
        recent_profits.append(latest_profit)
        
        # Keep rolling window of last 20 trades
        if len(recent_profits) > 20:
            recent_profits = recent_profits[-20:]
            wallet.performance.variance_metrics['recent_profits'] = recent_profits
        
        # Calculate variance if we have enough data
        if len(recent_profits) >= 5:
            profit_variance = np.var(recent_profits)
            wallet.performance.variance_metrics['profit_variance'] = profit_variance
            
            # Calculate other useful metrics
            wallet.performance.variance_metrics['profit_mean'] = np.mean(recent_profits)
            wallet.performance.variance_metrics['profit_std'] = np.std(recent_profits)
            
            # Calculate long-term performance metrics
            if len(recent_profits) >= 15:
                # Trend analysis (is performance improving over time?)
                first_half = recent_profits[:len(recent_profits)//2]
                second_half = recent_profits[len(recent_profits)//2:]
                
                trend_improvement = np.mean(second_half) - np.mean(first_half)
                wallet.performance.long_term_performance['trend_improvement'] = trend_improvement
                wallet.performance.long_term_performance['consistency_score'] = 1.0 / max(0.01, profit_variance)
    
    async def _update_wallet_tier_safe(self, wallet_id: str):
        """Update wallet tier based on performance - THREAD SAFE (called within performance lock)"""
        try:
            wallet = self.wallets[wallet_id]
            perf = wallet.performance
            old_tier = wallet.tier
            
            # Tier upgrade conditions (same logic, but atomic)
            if (perf.total_trades >= 50 and perf.win_rate >= 0.8 and 
                perf.total_profit_sol >= 1.0 and wallet.tier == WalletTier.ADVANCED):
                wallet.tier = WalletTier.MASTER
                
            elif (perf.total_trades >= 20 and perf.win_rate >= 0.7 and 
                  perf.total_profit_sol >= 0.5 and wallet.tier == WalletTier.INTERMEDIATE):
                wallet.tier = WalletTier.ADVANCED
                
            elif (perf.total_trades >= 10 and perf.win_rate >= 0.6 and 
                  perf.total_profit_sol >= 0.1 and wallet.tier == WalletTier.STARTER):
                wallet.tier = WalletTier.INTERMEDIATE
            
            # Log tier changes
            if wallet.tier != old_tier:
                self.logger.info(f"🏆 Wallet {wallet_id} upgraded: {old_tier.value} → {wallet.tier.value}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update wallet tier for {wallet_id}: {e}")
    
    async def _rotation_scheduler(self):
        """Background task to handle wallet rotation"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                if self.security_config.wallet_rotation_enabled:
                    await self._rotate_wallets_if_needed()
                
            except Exception as e:
                self.logger.error(f"❌ Rotation scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _rotate_wallets_if_needed(self):
        """Rotate wallets based on security schedule"""
        try:
            rotation_interval = timedelta(hours=self.security_config.wallet_rotation_interval_hours)
            current_time = datetime.now()
            
            for wallet_id, wallet in self.wallets.items():
                time_since_creation = current_time - wallet.created_at
                
                if time_since_creation >= rotation_interval and wallet.status == WalletStatus.ACTIVE:
                    # Check if wallet has high exposure
                    if wallet.balance_sol > self.security_config.max_wallet_exposure_sol:
                        await self._initiate_wallet_rotation(wallet_id)
            
        except Exception as e:
            self.logger.error(f"❌ Wallet rotation error: {e}")
    
    async def _initiate_wallet_rotation(self, wallet_id: str):
        """Initiate rotation for a specific wallet"""
        try:
            wallet = self.wallets[wallet_id]
            wallet.status = WalletStatus.ROTATING
            
            # Create replacement wallet
            new_tier = wallet.tier
            new_wallet_id = f"rotated_{wallet_id}_{secrets.token_hex(4)}"
            
            new_wallet = await self.create_wallet(new_wallet_id, new_tier)
            
            if new_wallet:
                # In production, we would transfer funds here
                self.logger.info(f"🔄 Initiated rotation: {wallet_id} -> {new_wallet_id}")
                
                # Retire old wallet after some time
                asyncio.create_task(self._retire_wallet_after_delay(wallet_id, 24))  # 24 hours
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rotate wallet {wallet_id}: {e}")
    
    async def _retire_wallet_after_delay(self, wallet_id: str, delay_hours: int):
        """Retire a wallet after specified delay"""
        await asyncio.sleep(delay_hours * 3600)
        
        if wallet_id in self.wallets:
            self.wallets[wallet_id].status = WalletStatus.RETIRED
            self.logger.info(f"🏁 Wallet {wallet_id} retired")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        active_wallets = [w for w in self.wallets.values() if w.status == WalletStatus.ACTIVE]
        
        tier_distribution = {}
        for tier in WalletTier:
            tier_distribution[tier.value] = len([w for w in active_wallets if w.tier == tier])
        
        total_balance = sum(w.balance_sol for w in active_wallets)
        
        return {
            'total_wallets': len(self.wallets),
            'active_wallets': len(active_wallets),
            'tier_distribution': tier_distribution,
            'total_balance_sol': total_balance,
            'rotation_enabled': self.security_config.wallet_rotation_enabled,
            'client_connected': self.client is not None
        }
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        try:
            self.logger.warning("🚨 Emergency wallet shutdown initiated")
            
            # Mark all wallets as emergency status
            for wallet in self.wallets.values():
                wallet.status = WalletStatus.EMERGENCY
            
            # Close Solana client
            if self.client:
                await self.client.close()
            
            self.logger.info("🛑 Emergency wallet shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency shutdown error: {e}")

    async def _evolution_scheduler(self):
        """Schedule nightly wallet evolution (from archived evolution_engine.py)"""
        while True:
            try:
                # Check if it's time for evolution cycle
                hours_since_last = (datetime.now() - self.last_evolution_cycle).total_seconds() / 3600
                
                if hours_since_last >= self.evolution_cycle_hours:
                    self.logger.info("🧬 Starting nightly wallet evolution cycle...")
                    await self._execute_evolution_cycle()
                    self.last_evolution_cycle = datetime.now()
                
                # Sleep for 1 hour between checks
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"❌ Evolution scheduler error: {e}")
                await asyncio.sleep(3600)
    
    async def _execute_evolution_cycle(self):
        """Execute nightly evolution cycle"""
        try:
            self.logger.info("🔄 Executing wallet evolution cycle...")
            
            # 1. Evaluate fitness of all wallets
            fitness_scores = await self._evaluate_wallet_fitness()
            
            # 2. Apply natural selection (retire poor performers)
            await self._natural_selection(fitness_scores)
            
            # 3. Breed successful wallets
            await self._breed_successful_wallets(fitness_scores)
            
            # 4. Apply genetic mutations
            await self._apply_genetic_mutations()
            
            # 5. Update signal trust based on performance
            await self._update_signal_trust()
            
            self.logger.info("✅ Evolution cycle completed")
            
        except Exception as e:
            self.logger.error(f"❌ Evolution cycle failed: {e}")
    
    async def _evaluate_wallet_fitness(self) -> Dict[str, float]:
        """Evaluate fitness scores for all wallets with enhanced delayed reward consideration"""
        fitness_scores = {}
        current_volatility = await self._calculate_market_volatility()
        
        for wallet_id, wallet in self.wallets.items():
            # Calculate fitness based on multiple factors
            performance = wallet.performance
            
            # Base fitness from profit and win rate
            profit_fitness = max(0, performance.total_profit_sol) * 0.3
            winrate_fitness = performance.win_rate * 0.25
            volume_fitness = min(performance.total_volume_sol / 100.0, 1.0) * 0.15
            consistency_fitness = (1.0 - performance.risk_score) * 0.1
            
            # NEW: Delayed reward consideration (accounts for trades that became profitable over time)
            delayed_reward_score = wallet.calculate_delayed_reward_score()
            delayed_fitness = delayed_reward_score * 0.15
            
            # NEW: Long-term variance tolerance (rewards stable long-term performance)
            variance_score = self._calculate_variance_score(wallet)
            variance_fitness = variance_score * 0.05
            
            total_fitness = (profit_fitness + winrate_fitness + volume_fitness + 
                           consistency_fitness + delayed_fitness + variance_fitness)
            
            # Update dynamic survival threshold based on market volatility and learning phase
            if wallet.learning_phase:
                performance.survival_threshold = max(0.1, 0.3 - current_volatility * 0.2)
            else:
                performance.survival_threshold = max(0.2, 0.3 + current_volatility * 0.1)
            
            fitness_scores[wallet_id] = total_fitness
            wallet.performance.fitness_score = total_fitness
            
        return fitness_scores
    
    async def _calculate_market_volatility(self) -> float:
        """Calculate current market volatility for dynamic threshold adjustment"""
        # Simplified volatility calculation - in production would use real market data
        recent_performances = [w.performance.risk_score for w in self.wallets.values()]
        if not recent_performances:
            return 0.5
        return min(1.0, np.std(recent_performances) * 2)
    
    def _calculate_variance_score(self, wallet: Wallet) -> float:
        """Calculate variance tolerance score for long-term performance stability"""
        if not wallet.performance.variance_metrics:
            return 0.5
            
        # Reward wallets that perform consistently over time
        profit_variance = wallet.performance.variance_metrics.get('profit_variance', 1.0)
        variance_tolerance = wallet.genetics.variance_tolerance
        
        # Lower variance is better, adjusted by genetic tolerance
        variance_score = max(0.0, 1.0 - profit_variance * (1.0 - variance_tolerance))
        return min(1.0, variance_score)
    
    async def _natural_selection(self, fitness_scores: Dict[str, float]):
        """Remove poor performing wallets with dynamic thresholds and wallet consensus"""
        sorted_wallets = sorted(fitness_scores.items(), key=lambda x: x[1])
        
        # NEW: Use dynamic survival thresholds instead of fixed percentages
        wallets_to_retire = []
        for wallet_id, fitness in sorted_wallets:
            wallet = self.wallets.get(wallet_id)
            if wallet and fitness < wallet.performance.survival_threshold:
                # NEW: Check if wallet consensus suggests keeping this wallet despite low fitness
                consensus_verdict = await self._get_wallet_consensus_verdict(wallet_id, fitness)
                if not consensus_verdict.should_keep:
                    wallets_to_retire.append((wallet_id, fitness))
        
        # Limit retirements to prevent system instability
        max_retirements = max(1, len(self.wallets) // 4)  # Max 25% per cycle
        wallets_to_retire = wallets_to_retire[:max_retirements]
        
        for wallet_id, fitness in wallets_to_retire:
            wallet = self.wallets.get(wallet_id)
            # NEW: Store experiment memory before retiring
            if wallet:
                await self._store_experiment_memory(wallet, "natural_selection_retirement")
            
            self.logger.info(f"🔥 Retiring wallet {wallet_id} (fitness: {fitness:.3f}, threshold: {wallet.performance.survival_threshold:.3f})")
            await self._retire_and_replace_wallet(wallet_id)
    
    async def _get_wallet_consensus_verdict(self, target_wallet_id: str, fitness: float) -> 'ConsensusVerdict':
        """Get consensus from other wallets about whether to keep a low-performing wallet"""
        from dataclasses import dataclass
        
        @dataclass
        class ConsensusVerdict:
            should_keep: bool
            confidence: float
            voting_wallets: int
            reasons: List[str]
        
        target_wallet = self.wallets.get(target_wallet_id)
        if not target_wallet:
            return ConsensusVerdict(False, 1.0, 0, ["Target wallet not found"])
        
        votes = []
        reasons = []
        
        # Get votes from top-performing wallets (they have more influence)
        top_wallets = sorted(self.wallets.items(), 
                           key=lambda x: x[1].performance.fitness_score, reverse=True)[:5]
        
        for wallet_id, wallet in top_wallets:
            if wallet_id == target_wallet_id:
                continue
                
            # Calculate vote based on:
            # 1. Similar genetics (if they share traits, might be worth keeping)
            # 2. Complementary performance (diversification value)
            # 3. Recent improvement trends
            
            genetic_similarity = self._calculate_genetic_similarity(wallet.genetics, target_wallet.genetics)
            
            # If genetics are similar to a high performer, give benefit of doubt
            if genetic_similarity > 0.7 and wallet.performance.fitness_score > 0.6:
                vote_strength = 0.8 * wallet.genetics.consensus_weight
                votes.append(vote_strength)
                reasons.append(f"Genetic similarity to high performer {wallet_id[:8]}")
            
            # Check if target wallet has unique valuable traits
            elif target_wallet.genetics.delayed_reward_sensitivity > 0.8:
                vote_strength = 0.6 * wallet.genetics.consensus_weight
                votes.append(vote_strength)
                reasons.append(f"High delayed reward sensitivity - rare trait")
            
            # Check recent improvement trends
            elif len(target_wallet.performance.delayed_rewards) > 0:
                recent_recoveries = sum(1 for dr in target_wallet.performance.delayed_rewards 
                                      if dr.is_profitable and dr.time_to_profit and dr.time_to_profit.days < 3)
                if recent_recoveries > 0:
                    vote_strength = 0.5 * wallet.genetics.consensus_weight
                    votes.append(vote_strength)
                    reasons.append(f"Recent recovery ability shown")
            else:
                # Default: retire the wallet
                votes.append(0.0)
        
        if not votes:
            return ConsensusVerdict(False, 1.0, 0, ["No voting wallets available"])
        
        avg_vote = sum(votes) / len(votes)
        should_keep = avg_vote > 0.5
        confidence = abs(avg_vote - 0.5) * 2
        
        return ConsensusVerdict(should_keep, confidence, len(votes), reasons)
    
    def _calculate_genetic_similarity(self, genetics1: EvolutionGenetics, genetics2: EvolutionGenetics) -> float:
        """Calculate similarity between two genetic profiles"""
        traits = ['aggression', 'risk_tolerance', 'patience', 'signal_trust',
                 'adaptation_rate', 'memory_strength', 'pattern_recognition',
                 'herd_immunity', 'leadership']
        
        similarities = []
        for trait in traits:
            val1 = getattr(genetics1, trait)
            val2 = getattr(genetics2, trait)
            similarity = 1.0 - abs(val1 - val2)  # Closer values = higher similarity
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    async def _store_experiment_memory(self, wallet: Wallet, reason: str):
        """Store failed experiment memory to avoid repetition"""
        # Get current market conditions (simplified)
        market_conditions = {
            'avg_fitness': sum(w.performance.fitness_score for w in self.wallets.values()) / len(self.wallets),
            'volatility': await self._calculate_market_volatility(),
            'wallet_count': len(self.wallets),
            'genetics_profile': {
                'aggression': wallet.genetics.aggression,
                'risk_tolerance': wallet.genetics.risk_tolerance,
                'patience': wallet.genetics.patience
            }
        }
        
        experiment_memory = ExperimentMemory(
            experiment_id=f"{wallet.id}_{datetime.now().isoformat()}",
            strategy_type=f"genetic_profile_{wallet.tier.value}",
            market_conditions=market_conditions,
            failure_reason=reason,
            failure_cost=abs(wallet.performance.total_profit_sol) if wallet.performance.total_profit_sol < 0 else 0,
            timestamp=datetime.now(),
            confidence_loss=1.0 - wallet.performance.fitness_score
        )
        
        # Store in all remaining wallets' memory
        for remaining_wallet in self.wallets.values():
            if remaining_wallet.id != wallet.id:
                remaining_wallet.performance.experiment_failures.append(experiment_memory)
                # Keep only recent memories (last 50)
                if len(remaining_wallet.performance.experiment_failures) > 50:
                    remaining_wallet.performance.experiment_failures = remaining_wallet.performance.experiment_failures[-50:]
    
    async def _breed_successful_wallets(self, fitness_scores: Dict[str, float]):
        """Create new wallets by breeding top performers"""
        sorted_wallets = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 30% can breed
        top_count = max(2, len(sorted_wallets) * 3 // 10)
        top_wallets = sorted_wallets[:top_count]
        
        for i in range(0, len(top_wallets), 2):
            if i + 1 < len(top_wallets):
                parent1_id = top_wallets[i][0]
                parent2_id = top_wallets[i + 1][0]
                
                # Create offspring with mixed genetics
                await self._create_genetic_offspring(parent1_id, parent2_id)
    
    async def _create_genetic_offspring(self, parent1_id: str, parent2_id: str):
        """Create offspring wallet with mixed genetics from parents"""
        parent1 = self.wallets[parent1_id]
        parent2 = self.wallets[parent2_id]
        
        # Mix genetics from both parents
        offspring_genetics = EvolutionGenetics()
        
        traits = ['aggression', 'risk_tolerance', 'patience', 'signal_trust',
                 'adaptation_rate', 'memory_strength', 'pattern_recognition',
                 'herd_immunity', 'leadership']
        
        for trait in traits:
            # Random mix from parents
            if random.random() < 0.5:
                value = getattr(parent1.genetics, trait)
            else:
                value = getattr(parent2.genetics, trait)
            
            # Add some variance
            variance = random.uniform(-0.1, 0.1)
            final_value = np.clip(value + variance, 0.0, 1.0)
            setattr(offspring_genetics, trait, final_value)
        
        # Create offspring wallet
        offspring_id = f"offspring_{parent1_id[:8]}_{parent2_id[:8]}_{secrets.token_hex(4)}"
        
        # Determine tier based on parent performance
        avg_fitness = (parent1.performance.fitness_score + parent2.performance.fitness_score) / 2
        if avg_fitness > 0.8:
            tier = WalletTier.MASTER
        elif avg_fitness > 0.6:
            tier = WalletTier.ADVANCED
        elif avg_fitness > 0.4:
            tier = WalletTier.INTERMEDIATE
        else:
            tier = WalletTier.STARTER
        
        self.logger.info(f"👶 Creating genetic offspring: {offspring_id}")
        # Note: In production, this would create the actual wallet when needed
        
    async def _apply_genetic_mutations(self):
        """Apply random mutations to wallet genetics"""
        for wallet in self.wallets.values():
            if random.random() < 0.3:  # 30% chance of mutation
                old_genetics = wallet.genetics
                wallet.genetics = old_genetics.mutate()
                self.logger.debug(f"🧬 Applied genetic mutation to {wallet.id}")
    
    async def _update_signal_trust(self):
        """Update signal trust for all wallets based on recent performance"""
        for wallet in self.wallets.values():
            wallet.signal_trust.daily_decay()
            
            # Adjust trust based on recent performance
            if wallet.performance.total_trades > 0:
                recent_success = wallet.performance.successful_trades / wallet.performance.total_trades
                
                # Boost signal trust for successful wallets
                if recent_success > 0.7:
                    for signal_type in ['technical_analysis', 'volume_analysis', 'liquidity_analysis']:
                        wallet.signal_trust.update_trust(signal_type, True, 0.1)
                        
    async def _retire_and_replace_wallet(self, wallet_id: str):
        """Retire a wallet and create a replacement"""
        if wallet_id in self.wallets:
            old_wallet = self.wallets[wallet_id]
            old_wallet.status = WalletStatus.RETIRED
            
            # Create replacement with evolved genetics
            replacement_id = f"evolved_{wallet_id}_{secrets.token_hex(4)}"
            evolved_genetics = old_wallet.genetics.mutate()
            
            await self.create_wallet(replacement_id, old_wallet.tier, evolved_genetics)

# Global wallet manager instance
_wallet_manager = None

async def get_wallet_manager() -> UnifiedWalletManager:
    """Get global wallet manager instance"""
    global _wallet_manager
    if _wallet_manager is None:
        _wallet_manager = UnifiedWalletManager()
        await _wallet_manager.initialize()
    return _wallet_manager 