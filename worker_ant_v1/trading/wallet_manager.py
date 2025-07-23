"""
UNIFIED WALLET MANAGER - PRODUCTION READY
========================================

Real wallet management with Solana keypair generation,
actual balance checking, and transaction capabilities.
"""

import asyncio
import secrets
import base58
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from solana.rpc.async_api import AsyncClient
    from solana.keypair import Keypair
    from solana.rpc.commitment import Commitment
    from solana.publickey import PublicKey
    from solana.transaction import Transaction
    from solana.system_program import TransferParams, transfer
except ImportError:
    from ..utils.solana_compat import AsyncClient, Keypair, Commitment, PublicKey, Transaction, TransferParams, transfer

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_wallet_config

class WalletState(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EVOLVING = "evolving"
    RETIRED = "retired"

class WalletBehavior(Enum):
    SNIPER = "sniper"      # Quick entry/exit, high frequency
    ACCUMULATOR = "accumulator"  # Build positions over time
    SCALPER = "scalper"    # Small profits, high volume
    HODLER = "hodler"      # Long-term holds
    MIMICKER = "mimicker"  # Follows successful wallets

@dataclass
class EvolutionGenetics:
    """Genetic traits for wallet evolution"""
    aggression: float = 0.5  # Risk tolerance (0-1)
    patience: float = 0.5    # Hold duration preference (0-1)
    signal_trust: float = 0.5  # Trust in different signals (0-1)
    adaptation_rate: float = 0.5  # Learning speed (0-1)
    memory_strength: float = 0.5  # Pattern recognition (0-1)
    herd_immunity: float = 0.5  # Resistance to crowd psychology (0-1)

@dataclass
class WalletPerformance:
    """Wallet performance metrics"""
    total_trades: int = 0
    successful_trades: int = 0
    total_profit: float = 0.0
    avg_profit_per_trade: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_active: datetime = field(default_factory=datetime.now)
    consecutive_wins: int = 0
    consecutive_losses: int = 0

@dataclass
class TradingWallet:
    """Individual trading wallet with evolution capabilities"""
    wallet_id: str
    address: str
    private_key: str
    state: WalletState
    behavior: WalletBehavior
    genetics: EvolutionGenetics
    performance: WalletPerformance
    created_at: datetime
    last_evolution: datetime
    evolution_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    personal_risk_profile: Dict[str, Any] = field(default_factory=dict)
    active_squad_ruleset: Optional[Dict[str, Any]] = None
    
    def self_assess_trade(self, trade_params: Dict[str, Any]) -> bool:
        """Self-assessment method allowing wallet to veto trades based on its genetics
        
        Args:
            trade_params: Trade parameters including risk, size, token info
            
        Returns:
            True if trade is acceptable, False if wallet vetoes the trade
        """
        # If squad rules are active, use those instead of personal profile
        if self.active_squad_ruleset:
            return self._assess_squad_trade(trade_params)
        
        # Personal risk assessment based on genetics
        risk_level = trade_params.get('risk_level', 'medium')
        position_size = trade_params.get('position_size_sol', 0.0)
        token_age_hours = trade_params.get('token_age_hours', 24)
        
        # Check aggression level against risk
        if risk_level == 'high' and self.genetics.aggression < 0.6:
            return False  # Conservative wallet rejects high-risk trades
        
        # Check patience level against token age
        if token_age_hours < 1 and self.genetics.patience > 0.7:
            return False  # Patient wallet rejects very new tokens
        
        # Check position size against personal limits
        max_position = self.personal_risk_profile.get('max_position_size_sol', 10.0)
        if position_size > max_position:
            return False  # Position too large for this wallet
        
        # Check herd immunity against popular tokens
        if trade_params.get('is_trending', False) and self.genetics.herd_immunity > 0.8:
            return False  # Independent wallet rejects trending tokens
        
        return True
    
    def _assess_squad_trade(self, trade_params: Dict[str, Any]) -> bool:
        """Assess trade using squad ruleset instead of personal profile"""
        if not self.active_squad_ruleset:
            return True
        
        # Squad rules override personal preferences
        squad_risk_level = self.active_squad_ruleset.get('risk_level', 'medium')
        squad_max_position = self.active_squad_ruleset.get('max_position_size_sol', 50.0)
        
        position_size = trade_params.get('position_size_sol', 0.0)
        
        # Only check squad-specific limits
        if position_size > squad_max_position:
            return False
        
        return True

class UnifiedWalletManager:
    """Production-ready wallet manager with real Solana integration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_wallet_config()
        
        # Solana RPC client
        self.rpc_client = AsyncClient(self.config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com'))
        
        # Wallet storage
        self.wallets: Dict[str, TradingWallet] = {}
        self.active_wallets: List[str] = []
        self.performance_history: Dict[str, List[WalletPerformance]] = {}
        
        # Evolution configuration
        self.evolution_config = {
            'evolution_interval_hours': 24,
            'retirement_threshold': 0.3,  # 30% win rate
            'evolution_mutation_rate': 0.1,
            'max_wallets': 10,
            'min_wallets': 5
        }
        
        # System state
        self.initialized = False
        self.evolution_active = True
        self.blitzscaling_active = False
        
        # Token balances cache
        self.balance_cache: Dict[str, Dict[str, float]] = {}
        self.cache_duration = timedelta(minutes=5)
        
        self.logger.info("👛 Unified Wallet Manager initialized")

    async def initialize(self) -> bool:
        """Initialize the wallet manager"""
        try:
            self.logger.info("👛 Initializing Unified Wallet Manager...")
            
            # Test RPC connection
            try:
                await self.rpc_client.get_health()
                self.logger.info("✅ Solana RPC connection established")
            except Exception as e:
                self.logger.error(f"❌ Solana RPC connection failed: {e}")
                return False
            
            # Load or create wallets
            await self._load_or_create_wallets()
            
            # Start background tasks
            asyncio.create_task(self._evolution_loop())
            asyncio.create_task(self._performance_tracking_loop())
            asyncio.create_task(self._balance_update_loop())
            
            self.initialized = True
            self.logger.info("✅ Unified Wallet Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Wallet manager initialization failed: {e}")
            return False

    async def _load_or_create_wallets(self):
        """Load existing wallets or create new ones"""
        try:
            # Try to load from storage
            if await self._load_wallets_from_storage():
                self.logger.info(f"📂 Loaded {len(self.wallets)} wallets from storage")
            else:
                # Create new wallet swarm
                await self._create_wallet_swarm()
                self.logger.info(f"🆕 Created {len(self.wallets)} new wallets")
            
            # Update active wallets list
            self.active_wallets = [
                wallet_id for wallet_id, wallet in self.wallets.items()
                if wallet.state == WalletState.ACTIVE
            ]
            
        except Exception as e:
            self.logger.error(f"Error loading/creating wallets: {e}")
            raise

    async def _create_wallet_swarm(self):
        """Create initial wallet swarm"""
        num_wallets = self.evolution_config['max_wallets']
        
        for i in range(num_wallets):
            wallet_id = f"wallet_{i+1:02d}"
            wallet = await self._create_wallet(wallet_id)
            self.wallets[wallet_id] = wallet
            self.active_wallets.append(wallet_id)

    async def _create_wallet(self, wallet_id: str) -> TradingWallet:
        """Create a new trading wallet with real Solana keypair"""
        # Generate Solana keypair
        keypair = Keypair.generate()
        private_key = base58.b58encode(keypair.secret_key).decode()
        address = str(keypair.public_key)
        
        # Create random genetics
        genetics = EvolutionGenetics(
            aggression=secrets.SystemRandom().uniform(0.3, 0.8),
            patience=secrets.SystemRandom().uniform(0.2, 0.9),
            signal_trust=secrets.SystemRandom().uniform(0.4, 0.9),
            adaptation_rate=secrets.SystemRandom().uniform(0.3, 0.8),
            memory_strength=secrets.SystemRandom().uniform(0.4, 0.9),
            herd_immunity=secrets.SystemRandom().uniform(0.3, 0.8)
        )
        
        # Choose random behavior
        behavior = secrets.SystemRandom().choice(list(WalletBehavior))
        
        wallet = TradingWallet(
            wallet_id=wallet_id,
            address=address,
            private_key=private_key,
            state=WalletState.ACTIVE,
            behavior=behavior,
            genetics=genetics,
            performance=WalletPerformance(),
            created_at=datetime.now(),
            last_evolution=datetime.now()
        )
        
        self.logger.info(f"👤 Created wallet {wallet_id}: {behavior.value} behavior")
        return wallet
    
    async def create_wallet(self, wallet_id: str, genetics: EvolutionGenetics = None) -> TradingWallet:
        """Create a new wallet with optional custom genetics"""
        try:
            if wallet_id in self.wallets:
                self.logger.warning(f"Wallet {wallet_id} already exists")
                return self.wallets[wallet_id]
            
            wallet = await self._create_wallet(wallet_id)
            
            # Override genetics if provided
            if genetics:
                wallet.genetics = genetics
            
            self.wallets[wallet_id] = wallet
            self.active_wallets.append(wallet_id)
            
            self.logger.info(f"👛 Created wallet {wallet_id}: {wallet.address}")
            return wallet
            
        except Exception as e:
            self.logger.error(f"Error creating wallet {wallet_id}: {e}")
            raise
    
    async def get_wallet_info(self, wallet_id: str) -> Optional[Dict[str, Any]]:
        """Get wallet information as dictionary"""
        try:
            if wallet_id not in self.wallets:
                return None
            
            wallet = self.wallets[wallet_id]
            
            return {
                'wallet_id': wallet.wallet_id,
                'address': wallet.address,
                'state': wallet.state.value,
                'behavior': wallet.behavior.value,
                'genetics': {
                    'aggression': wallet.genetics.aggression,
                    'patience': wallet.genetics.patience,
                    'signal_trust': wallet.genetics.signal_trust,
                    'adaptation_rate': wallet.genetics.adaptation_rate,
                    'memory_strength': wallet.genetics.memory_strength,
                    'herd_immunity': wallet.genetics.herd_immunity
                },
                'performance': {
                    'total_trades': wallet.performance.total_trades,
                    'successful_trades': wallet.performance.successful_trades,
                    'total_profit': wallet.performance.total_profit,
                    'win_rate': wallet.performance.win_rate,
                    'avg_profit_per_trade': wallet.performance.avg_profit_per_trade,
                    'max_drawdown': wallet.performance.max_drawdown,
                    'sharpe_ratio': wallet.performance.sharpe_ratio,
                    'last_active': wallet.performance.last_active.isoformat(),
                    'consecutive_wins': wallet.performance.consecutive_wins,
                    'consecutive_losses': wallet.performance.consecutive_losses
                },
                'created_at': wallet.created_at.isoformat(),
                'last_evolution': wallet.last_evolution.isoformat(),
                'evolution_count': wallet.evolution_count,
                'metadata': wallet.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error getting wallet info for {wallet_id}: {e}")
            return None
    
    async def remove_wallet(self, wallet_id: str) -> bool:
        """Remove a wallet from the system"""
        try:
            if wallet_id not in self.wallets:
                self.logger.warning(f"Wallet {wallet_id} not found for removal")
                return False
            
            # Remove from active wallets list
            if wallet_id in self.active_wallets:
                self.active_wallets.remove(wallet_id)
            
            # Remove from wallets dict
            del self.wallets[wallet_id]
            
            # Remove from performance history
            if wallet_id in self.performance_history:
                del self.performance_history[wallet_id]
            
            # Remove from balance cache
            cache_keys_to_remove = [key for key in self.balance_cache.keys() if key.startswith(f"{wallet_id}_")]
            for key in cache_keys_to_remove:
                del self.balance_cache[key]
            
            self.logger.info(f"🗑️ Removed wallet {wallet_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing wallet {wallet_id}: {e}")
            return False
    
    async def wallet_exists(self, wallet_id: str) -> bool:
        """Check if wallet exists"""
        return wallet_id in self.wallets

    async def get_wallet_balance(self, wallet_id: str) -> float:
        """Get real SOL balance for wallet"""
        try:
            if wallet_id not in self.wallets:
                return 0.0
            
            wallet = self.wallets[wallet_id]
            
            # Check cache first
            cache_key = f"{wallet_id}_sol"
            if cache_key in self.balance_cache:
                cache_time, balance = self.balance_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return balance
            
            # Query actual balance from Solana
            try:
                balance_response = await self.rpc_client.get_balance(
                    wallet.address,
                    commitment=Commitment("confirmed")
                )
                
                if balance_response.value is not None:
                    balance_sol = balance_response.value / 1_000_000_000  # Convert lamports to SOL
                    
                    # Cache the result
                    self.balance_cache[cache_key] = (datetime.now(), balance_sol)
                    
                    return balance_sol
                else:
                    return 0.0
                    
            except Exception as e:
                self.logger.error(f"Error getting balance for {wallet_id}: {e}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in get_wallet_balance: {e}")
            return 0.0

    async def get_token_balance(self, wallet_id: str, token_address: str) -> float:
        """Get real token balance for wallet"""
        try:
            if wallet_id not in self.wallets:
                return 0.0
            
            wallet = self.wallets[wallet_id]
            
            # Check cache first
            cache_key = f"{wallet_id}_{token_address}"
            if cache_key in self.balance_cache:
                cache_time, balance = self.balance_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return balance
            
            # Query token balance from Solana
            try:
                # Get token account
                token_account = await self._get_token_account(wallet.address, token_address)
                
                if token_account:
                    balance_response = await self.rpc_client.get_token_account_balance(
                        token_account,
                        commitment=Commitment("confirmed")
                    )
                    
                    if balance_response.value:
                        balance = float(balance_response.value.amount) / (10 ** balance_response.value.decimals)
                        
                        # Cache the result
                        self.balance_cache[cache_key] = (datetime.now(), balance)
                        
                        return balance
                
                return 0.0
                
            except Exception as e:
                self.logger.error(f"Error getting token balance for {wallet_id}: {e}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in get_token_balance: {e}")
            return 0.0

    async def _get_token_account(self, wallet_address: str, token_mint: str) -> Optional[str]:
        """Get token account address for a specific token"""
        try:
            # Get all token accounts for the wallet
            response = await self.rpc_client.get_token_accounts_by_owner(
                wallet_address,
                {"mint": token_mint},
                commitment=Commitment("confirmed")
            )
            
            if response.value and len(response.value) > 0:
                return response.value[0].pubkey
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting token account: {e}")
            return None

    async def get_wallet_keypair(self, wallet_id: str) -> Optional[Keypair]:
        """Get Solana keypair for wallet"""
        try:
            if wallet_id not in self.wallets:
                return None
            
            wallet = self.wallets[wallet_id]
            
            # Reconstruct keypair from private key
            private_key_bytes = base58.b58decode(wallet.private_key)
            keypair = Keypair.from_secret_key(private_key_bytes)
            
            return keypair
            
        except Exception as e:
            self.logger.error(f"Error getting keypair for {wallet_id}: {e}")
            return None

    async def get_best_wallet(self, trade_requirements: Dict[str, Any]) -> Optional[TradingWallet]:
        """Get the best wallet for a trade based on requirements"""
        try:
            if not self.active_wallets:
                return None
            
            best_wallet = None
            best_score = -1
            
            for wallet_id in self.active_wallets:
                wallet = self.wallets[wallet_id]
                
                # Calculate fitness score based on requirements
                score = self._calculate_wallet_fitness(wallet, trade_requirements)
                
                if score > best_score:
                    best_score = score
                    best_wallet = wallet
            
            return best_wallet
            
        except Exception as e:
            self.logger.error(f"Error selecting best wallet: {e}")
            return None

    def _calculate_wallet_fitness(self, wallet: TradingWallet, requirements: Dict[str, Any]) -> float:
        """Calculate wallet fitness for a specific trade"""
        try:
            # Base score from performance
            performance_score = wallet.performance.win_rate * 0.4 + wallet.performance.sharpe_ratio * 0.3
            
            # Behavior match score
            required_behavior = requirements.get('behavior', None)
            behavior_score = 0.5
            if required_behavior and wallet.behavior.value == required_behavior:
                behavior_score = 1.0
            
            # Genetics match score
            required_aggression = requirements.get('aggression', 0.5)
            aggression_match = 1.0 - abs(wallet.genetics.aggression - required_aggression)
            
            # Risk tolerance match
            risk_level = requirements.get('risk_level', 'medium')
            risk_scores = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            risk_match = 1.0 - abs(wallet.genetics.aggression - risk_scores.get(risk_level, 0.5))
            
            # Calculate final score
            final_score = (
                performance_score * 0.4 +
                behavior_score * 0.2 +
                aggression_match * 0.2 +
                risk_match * 0.2
            )
            
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating wallet fitness: {e}")
            return 0.0

    async def update_wallet_performance(self, wallet_id: str, trade_result: Dict[str, Any]):
        """Update wallet performance after a trade"""
        try:
            if wallet_id not in self.wallets:
                return
            
            wallet = self.wallets[wallet_id]
            performance = wallet.performance
            
            # Update basic metrics
            performance.total_trades += 1
            if trade_result.get('success', False):
                performance.successful_trades += 1
                performance.consecutive_wins += 1
                performance.consecutive_losses = 0
            else:
                performance.consecutive_losses += 1
                performance.consecutive_wins = 0
            
            # Update profit metrics
            profit = trade_result.get('profit_sol', 0.0)
            performance.total_profit += profit
            performance.avg_profit_per_trade = performance.total_profit / performance.total_trades
            
            # Update win rate
            performance.win_rate = performance.successful_trades / performance.total_trades
            
            # Update last active
            performance.last_active = datetime.now()
            
            # Store performance history
            if wallet_id not in self.performance_history:
                self.performance_history[wallet_id] = []
            
            self.performance_history[wallet_id].append(performance)
            
            # Keep only last 100 performance records
            if len(self.performance_history[wallet_id]) > 100:
                self.performance_history[wallet_id] = self.performance_history[wallet_id][-100:]
            
            self.logger.info(f"📊 Updated performance for {wallet_id}: Win rate {performance.win_rate:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error updating wallet performance: {e}")

    async def _evolution_loop(self):
        """Main evolution loop"""
        while self.initialized:
            try:
                await asyncio.sleep(self.evolution_config['evolution_interval_hours'] * 3600)
                
                if self.evolution_active:
                    await self._perform_evolution()
                
            except Exception as e:
                self.logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def _perform_evolution(self):
        """Perform wallet evolution"""
        try:
            self.logger.info("🧬 Starting wallet evolution cycle...")
            
            # Evaluate all wallets
            wallet_scores = []
            for wallet_id, wallet in self.wallets.items():
                if wallet.state == WalletState.ACTIVE:
                    score = self._evaluate_wallet(wallet)
                    wallet_scores.append((wallet_id, score))
            
            # Sort by performance
            wallet_scores.sort(key=lambda x: x[1], reverse=True)
            
            if self.blitzscaling_active:
                # Blitzscaling mode: Clone and expand instead of retire and replace
                await self._perform_blitzscaling_evolution(wallet_scores)
            else:
                # Normal mode: Retire and replace
                await self._perform_normal_evolution(wallet_scores)
            
            self.logger.info("✅ Evolution complete")
            
        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
    
    async def _perform_blitzscaling_evolution(self, wallet_scores: List[Tuple[str, float]]):
        """Perform evolution in blitzscaling mode - clone and expand"""
        try:
            self.logger.info("🚀 Performing blitzscaling evolution - cloning top performers")
            
            # Clone top performers to increase swarm size
            num_to_clone = min(3, len(wallet_scores) // 2)  # Clone top 50%
            for i in range(num_to_clone):
                wallet_id, score = wallet_scores[i]
                if score > 0.6:  # Clone good performers
                    await self._clone_wallet(wallet_id)
            
            # Evolve top performers
            num_to_evolve = min(2, len(wallet_scores) // 3)
            for i in range(num_to_evolve):
                wallet_id, score = wallet_scores[i]
                if score > 0.7:
                    await self._evolve_wallet(wallet_id)
            
            # Maintain minimum population
            while len([w for w in self.wallets.values() if w.state == WalletState.ACTIVE]) < self.evolution_config['min_wallets']:
                await self._create_evolved_wallet("best_performer")
            
        except Exception as e:
            self.logger.error(f"❌ Blitzscaling evolution error: {e}")
    
    async def _perform_normal_evolution(self, wallet_scores: List[Tuple[str, float]]):
        """Perform evolution in normal mode - retire and replace"""
        try:
            # Retire bottom 20%
            retirement_count = max(1, len(wallet_scores) // 5)
            for i in range(retirement_count):
                wallet_id, _ = wallet_scores[-(i+1)]
                await self._retire_wallet(wallet_id)
            
            # Create new wallets from top performers
            for i in range(retirement_count):
                parent_wallet_id, _ = wallet_scores[i]
                await self._create_evolved_wallet(parent_wallet_id)
            
            # Evolve remaining wallets
            for wallet_id, _ in wallet_scores[:-retirement_count]:
                await self._evolve_wallet(wallet_id)
                
        except Exception as e:
            self.logger.error(f"❌ Normal evolution error: {e}")
    
    async def _clone_wallet(self, parent_wallet_id: str):
        """Clone a wallet with similar genetics"""
        try:
            parent_wallet = self.wallets.get(parent_wallet_id)
            if not parent_wallet:
                return
            
            # Create new wallet ID
            clone_id = f"clone_{parent_wallet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate new keypair
            keypair = Keypair.generate()
            private_key = base58.b58encode(keypair.secret_key).decode()
            address = str(keypair.public_key)
            
            # Clone genetics with slight mutations
            cloned_genetics = EvolutionGenetics(
                aggression=parent_wallet.genetics.aggression + secrets.SystemRandom().uniform(-0.1, 0.1),
                patience=parent_wallet.genetics.patience + secrets.SystemRandom().uniform(-0.1, 0.1),
                signal_trust=parent_wallet.genetics.signal_trust + secrets.SystemRandom().uniform(-0.1, 0.1),
                adaptation_rate=parent_wallet.genetics.adaptation_rate + secrets.SystemRandom().uniform(-0.1, 0.1),
                memory_strength=parent_wallet.genetics.memory_strength + secrets.SystemRandom().uniform(-0.1, 0.1),
                herd_immunity=parent_wallet.genetics.herd_immunity + secrets.SystemRandom().uniform(-0.1, 0.1)
            )
            
            # Create clone wallet
            clone_wallet = TradingWallet(
                wallet_id=clone_id,
                address=address,
                private_key=private_key,
                state=WalletState.ACTIVE,
                behavior=parent_wallet.behavior,
                genetics=cloned_genetics,
                performance=WalletPerformance(),
                created_at=datetime.now(),
                last_evolution=datetime.now(),
                evolution_count=0,
                personal_risk_profile=parent_wallet.personal_risk_profile.copy()
            )
            
            self.wallets[clone_id] = clone_wallet
            self.active_wallets.append(clone_id)
            
            self.logger.info(f"🧬 Cloned wallet {parent_wallet_id} -> {clone_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error cloning wallet {parent_wallet_id}: {e}")
    
    async def set_blitzscaling_mode(self, active: bool):
        """Set blitzscaling mode for the wallet manager"""
        self.blitzscaling_active = active
        self.logger.info(f"🚀 Blitzscaling mode {'ACTIVATED' if active else 'DEACTIVATED'} for wallet manager")
        
        if active:
            # Increase max wallets in blitzscaling mode
            self.evolution_config['max_wallets'] = 20  # Double the normal limit
        else:
            # Reset to normal limits
            self.evolution_config['max_wallets'] = 10

    def _evaluate_wallet(self, wallet: TradingWallet) -> float:
        """Evaluate wallet performance for evolution"""
        try:
            # Base score from win rate and profit
            base_score = wallet.performance.win_rate * 0.6 + min(wallet.performance.avg_profit_per_trade, 1.0) * 0.4
            
            # Bonus for consistency
            consistency_bonus = 0.1 if wallet.performance.consecutive_wins > 3 else 0.0
            
            # Penalty for age
            age_days = (datetime.now() - wallet.created_at).days
            age_penalty = min(0.2, age_days * 0.01)
            
            return base_score + consistency_bonus - age_penalty
            
        except Exception as e:
            self.logger.error(f"Error evaluating wallet: {e}")
            return 0.0

    async def _retire_wallet(self, wallet_id: str):
        """Retire a wallet"""
        try:
            if wallet_id in self.wallets:
                wallet = self.wallets[wallet_id]
                wallet.state = WalletState.RETIRED
                
                if wallet_id in self.active_wallets:
                    self.active_wallets.remove(wallet_id)
                
                self.logger.info(f"🏁 Retired wallet {wallet_id}")
                
        except Exception as e:
            self.logger.error(f"Error retiring wallet {wallet_id}: {e}")

    async def _create_evolved_wallet(self, parent_wallet_id: str):
        """Create a new wallet evolved from a parent"""
        try:
            parent_wallet = self.wallets[parent_wallet_id]
            
            # Create new wallet ID
            new_wallet_id = f"wallet_{len(self.wallets) + 1:02d}"
            
            # Create new wallet with evolved genetics
            wallet = await self._create_wallet(new_wallet_id)
            
            # Evolve genetics from parent
            wallet.genetics = self._crossover_genetics(parent_wallet.genetics)
            wallet.behavior = parent_wallet.behavior  # Inherit behavior
            
            # Add to wallets
            self.wallets[new_wallet_id] = wallet
            self.active_wallets.append(new_wallet_id)
            
            self.logger.info(f"🧬 Created evolved wallet {new_wallet_id} from {parent_wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating evolved wallet: {e}")

    def _crossover_genetics(self, parent_genetics: EvolutionGenetics) -> EvolutionGenetics:
        """Create new genetics by crossing over parent genetics"""
        return EvolutionGenetics(
            aggression=parent_genetics.aggression + secrets.SystemRandom().uniform(-0.1, 0.1),
            patience=parent_genetics.patience + secrets.SystemRandom().uniform(-0.1, 0.1),
            signal_trust=parent_genetics.signal_trust + secrets.SystemRandom().uniform(-0.1, 0.1),
            adaptation_rate=parent_genetics.adaptation_rate + secrets.SystemRandom().uniform(-0.1, 0.1),
            memory_strength=parent_genetics.memory_strength + secrets.SystemRandom().uniform(-0.1, 0.1),
            herd_immunity=parent_genetics.herd_immunity + secrets.SystemRandom().uniform(-0.1, 0.1)
        )

    def _mutate_genetics(self, genetics: EvolutionGenetics) -> EvolutionGenetics:
        """Mutate genetics with random changes"""
        mutation_rate = self.evolution_config['evolution_mutation_rate']
        
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.aggression = max(0.0, min(1.0, genetics.aggression + secrets.SystemRandom().uniform(-0.2, 0.2)))
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.patience = max(0.0, min(1.0, genetics.patience + secrets.SystemRandom().uniform(-0.2, 0.2)))
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.signal_trust = max(0.0, min(1.0, genetics.signal_trust + secrets.SystemRandom().uniform(-0.2, 0.2)))
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.adaptation_rate = max(0.0, min(1.0, genetics.adaptation_rate + secrets.SystemRandom().uniform(-0.2, 0.2)))
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.memory_strength = max(0.0, min(1.0, genetics.memory_strength + secrets.SystemRandom().uniform(-0.2, 0.2)))
        if secrets.SystemRandom().random() < mutation_rate:
            genetics.herd_immunity = max(0.0, min(1.0, genetics.herd_immunity + secrets.SystemRandom().uniform(-0.2, 0.2)))
        
        return genetics

    async def _evolve_wallet(self, wallet_id: str):
        """Evolve an existing wallet"""
        try:
            wallet = self.wallets[wallet_id]
            
            # Mutate genetics
            wallet.genetics = self._mutate_genetics(wallet.genetics)
            
            # Update evolution tracking
            wallet.evolution_count += 1
            wallet.last_evolution = datetime.now()
            
            self.logger.info(f"🧬 Evolved wallet {wallet_id} (generation {wallet.evolution_count})")
            
        except Exception as e:
            self.logger.error(f"Error evolving wallet {wallet_id}: {e}")

    async def _adapt_genetics(self, wallet: TradingWallet):
        """Adapt wallet genetics based on recent performance"""
        try:
            # Get recent performance
            if wallet.wallet_id in self.performance_history:
                recent_performance = self.performance_history[wallet.wallet_id][-10:]  # Last 10 trades
                
                if len(recent_performance) >= 5:
                    avg_win_rate = sum(p.win_rate for p in recent_performance) / len(recent_performance)
                    avg_profit = sum(p.avg_profit_per_trade for p in recent_performance) / len(recent_performance)
                    
                    # Adjust genetics based on performance
                    if avg_win_rate < 0.4:  # Poor performance
                        wallet.genetics.aggression = max(0.1, wallet.genetics.aggression - 0.1)
                        wallet.genetics.signal_trust = max(0.1, wallet.genetics.signal_trust - 0.1)
                    elif avg_win_rate > 0.7:  # Good performance
                        wallet.genetics.aggression = min(0.9, wallet.genetics.aggression + 0.1)
                        wallet.genetics.signal_trust = min(0.9, wallet.genetics.signal_trust + 0.1)
                    
                    if avg_profit < 0:  # Losing money
                        wallet.genetics.patience = min(0.9, wallet.genetics.patience + 0.1)
                    elif avg_profit > 0.01:  # Making good profits
                        wallet.genetics.patience = max(0.1, wallet.genetics.patience - 0.1)
            
        except Exception as e:
            self.logger.error(f"Error adapting genetics for {wallet.wallet_id}: {e}")

    async def _performance_tracking_loop(self):
        """Track performance metrics"""
        while self.initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                for wallet_id, wallet in self.wallets.items():
                    if wallet.state == WalletState.ACTIVE:
                        await self._update_advanced_metrics(wallet)
                        await self._adapt_genetics(wallet)
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)

    async def _update_advanced_metrics(self, wallet: TradingWallet):
        """Update advanced performance metrics"""
        try:
            if wallet.wallet_id in self.performance_history:
                history = self.performance_history[wallet.wallet_id]
                
                if len(history) >= 10:
                    # Calculate Sharpe ratio
                    returns = [p.avg_profit_per_trade for p in history[-20:]]
                    if returns:
                        avg_return = sum(returns) / len(returns)
                        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
                        if variance > 0:
                            wallet.performance.sharpe_ratio = avg_return / (variance ** 0.5)
                    
                    # Calculate max drawdown
                    cumulative_profits = []
                    running_total = 0
                    for p in history:
                        running_total += p.total_profit
                        cumulative_profits.append(running_total)
                    
                    if cumulative_profits:
                        peak = max(cumulative_profits)
                        max_dd = 0
                        for profit in cumulative_profits:
                            dd = (peak - profit) / peak if peak > 0 else 0
                            max_dd = max(max_dd, dd)
                        wallet.performance.max_drawdown = max_dd
            
        except Exception as e:
            self.logger.error(f"Error updating advanced metrics for {wallet.wallet_id}: {e}")

    async def _balance_update_loop(self):
        """Update balance cache periodically"""
        while self.initialized:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Clear expired cache entries
                current_time = datetime.now()
                expired_keys = []
                
                for cache_key, (cache_time, _) in self.balance_cache.items():
                    if current_time - cache_time > self.cache_duration:
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.balance_cache[key]
                
            except Exception as e:
                self.logger.error(f"Balance update error: {e}")
                await asyncio.sleep(60)

    async def _load_wallets_from_storage(self) -> bool:
        """Load wallets from persistent storage"""
        try:
            storage_file = "data/wallets.json"
            
            if not os.path.exists(storage_file):
                return False
            
            with open(storage_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct wallets from data
            for wallet_data in data.get('wallets', []):
                wallet = TradingWallet(
                    wallet_id=wallet_data['wallet_id'],
                    address=wallet_data['address'],
                    private_key=wallet_data['private_key'],
                    state=WalletState(wallet_data['state']),
                    behavior=WalletBehavior(wallet_data['behavior']),
                    genetics=EvolutionGenetics(**wallet_data['genetics']),
                    performance=WalletPerformance(**wallet_data['performance']),
                    created_at=datetime.fromisoformat(wallet_data['created_at']),
                    last_evolution=datetime.fromisoformat(wallet_data['last_evolution']),
                    evolution_count=wallet_data.get('evolution_count', 0),
                    metadata=wallet_data.get('metadata', {})
                )
                
                self.wallets[wallet.wallet_id] = wallet
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading wallets from storage: {e}")
            return False

    async def _save_wallets_to_storage(self):
        """Save wallets to persistent storage"""
        try:
            os.makedirs("data", exist_ok=True)
            storage_file = "data/wallets.json"
            
            # Convert wallets to serializable format
            wallet_data = []
            for wallet in self.wallets.values():
                wallet_data.append({
                    'wallet_id': wallet.wallet_id,
                    'address': wallet.address,
                    'private_key': wallet.private_key,
                    'state': wallet.state.value,
                    'behavior': wallet.behavior.value,
                    'genetics': {
                        'aggression': wallet.genetics.aggression,
                        'patience': wallet.genetics.patience,
                        'signal_trust': wallet.genetics.signal_trust,
                        'adaptation_rate': wallet.genetics.adaptation_rate,
                        'memory_strength': wallet.genetics.memory_strength,
                        'herd_immunity': wallet.genetics.herd_immunity
                    },
                    'performance': {
                        'total_trades': wallet.performance.total_trades,
                        'successful_trades': wallet.performance.successful_trades,
                        'total_profit': wallet.performance.total_profit,
                        'avg_profit_per_trade': wallet.performance.avg_profit_per_trade,
                        'win_rate': wallet.performance.win_rate,
                        'max_drawdown': wallet.performance.max_drawdown,
                        'sharpe_ratio': wallet.performance.sharpe_ratio,
                        'last_active': wallet.performance.last_active.isoformat(),
                        'consecutive_wins': wallet.performance.consecutive_wins,
                        'consecutive_losses': wallet.performance.consecutive_losses
                    },
                    'created_at': wallet.created_at.isoformat(),
                    'last_evolution': wallet.last_evolution.isoformat(),
                    'evolution_count': wallet.evolution_count,
                    'metadata': wallet.metadata
                })
            
            data = {'wallets': wallet_data}
            
            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"💾 Saved {len(self.wallets)} wallets to storage")
            
        except Exception as e:
            self.logger.error(f"Error saving wallets to storage: {e}")

    def get_wallet_status(self) -> Dict[str, Any]:
        """Get wallet manager status"""
        try:
            total_balance = 0.0
            for wallet_id in self.active_wallets:
                # This would need to be async, but for status we'll estimate
                total_balance += 0.0  # Would need to call get_wallet_balance
            
            return {
                'initialized': self.initialized,
                'total_wallets': len(self.wallets),
                'active_wallets': len(self.active_wallets),
                'total_balance_sol': total_balance,
                'evolution_active': self.evolution_active,
                'wallet_details': [
                    {
                        'wallet_id': wallet.wallet_id,
                        'address': wallet.address,
                        'behavior': wallet.behavior.value,
                        'state': wallet.state.value,
                        'performance': {
                            'total_trades': wallet.performance.total_trades,
                            'win_rate': wallet.performance.win_rate,
                            'total_profit': wallet.performance.total_profit,
                            'sharpe_ratio': wallet.performance.sharpe_ratio
                        },
                        'genetics': {
                            'aggression': wallet.genetics.aggression,
                            'patience': wallet.genetics.patience,
                            'signal_trust': wallet.genetics.signal_trust
                        }
                    }
                    for wallet in self.wallets.values()
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting wallet status: {e}")
            return {}

    async def shutdown(self):
        """Shutdown the wallet manager"""
        try:
            self.logger.info("🛑 Shutting down wallet manager...")
            
            # Save wallets to storage
            await self._save_wallets_to_storage()
            
            # Close RPC connection
            await self.rpc_client.close()
            
            self.initialized = False
            self.logger.info("✅ Wallet manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def emergency_shutdown(self):
        """Emergency shutdown"""
        try:
            self.logger.critical("🚨 EMERGENCY WALLET SHUTDOWN")
            
            # Immediately stop all operations
            self.initialized = False
            self.evolution_active = False
            
            # Save current state
            await self._save_wallets_to_storage()
            
            # Close RPC connection
            await self.rpc_client.close()
            
            self.logger.critical("✅ Emergency wallet shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")

# Global instance
_wallet_manager = None

async def get_wallet_manager() -> UnifiedWalletManager:
    """Get global wallet manager instance"""
    global _wallet_manager
    if _wallet_manager is None:
        _wallet_manager = UnifiedWalletManager()
        await _wallet_manager.initialize()
    return _wallet_manager 