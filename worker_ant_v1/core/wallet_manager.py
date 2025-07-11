"""
UNIFIED WALLET MANAGER - 10-WALLET SWARM SYSTEM
==============================================

Manages 10 evolving trading wallets with genetic algorithms,
performance tracking, and autonomous evolution.
"""

import asyncio
import secrets
import hashlib
import base64
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path

from worker_ant_v1.utils.logger import setup_logger
from solana.keypair import Keypair
import base58

class WalletState(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    EVOLVING = "evolving"
    RETIRED = "retired"

class WalletBehavior(Enum):
    SNIPER = "sniper"
    ACCUMULATOR = "accumulator"
    SCALPER = "scalper"
    HODLER = "hodler"
    MIMICKER = "mimicker"

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

class UnifiedWalletManager:
    """Manages 10 evolving trading wallets"""
    
    def __init__(self):
        self.logger = setup_logger("UnifiedWalletManager")
        
        # Wallet storage
        self.wallets: Dict[str, TradingWallet] = {}
        self.active_wallets: List[str] = []
        self.retired_wallets: List[str] = []
        
        # Evolution settings
        self.evolution_config = {
            'evolution_interval_hours': 24,
            'min_performance_threshold': 0.6,
            'max_wallet_age_days': 7,
            'evolution_mutation_rate': 0.1,
            'crossover_rate': 0.7
        }
        
        # Performance tracking
        self.performance_history: Dict[str, List[WalletPerformance]] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        
        # System state
        self.initialized = False
        self.evolution_active = False
        
    async def initialize(self) -> bool:
        """Initialize the wallet manager"""
        try:
            self.logger.info("💰 Initializing Unified Wallet Manager...")
            
            # Load existing wallets or create new ones
            await self._load_or_create_wallets()
            
            # Start evolution system
            asyncio.create_task(self._evolution_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.initialized = True
            self.logger.info(f"✅ Wallet manager initialized with {len(self.wallets)} wallets")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize wallet manager: {e}")
            return False
    
    async def _load_or_create_wallets(self):
        """Load existing wallets or create new ones"""
        try:
            # Try to load from storage
            if await self._load_wallets_from_storage():
                self.logger.info("📁 Loaded existing wallets from storage")
            else:
                # Create new wallet swarm
                await self._create_wallet_swarm()
                self.logger.info("🆕 Created new wallet swarm")
            
            # Update active wallets list
            self.active_wallets = [
                wallet_id for wallet_id, wallet in self.wallets.items()
                if wallet.state == WalletState.ACTIVE
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to load/create wallets: {e}")
            # Create basic wallet swarm as fallback
            await self._create_wallet_swarm()
    
    async def _create_wallet_swarm(self):
        """Create 10 new trading wallets"""
        for i in range(10):
            wallet = await self._create_wallet(f"wallet_{i+1}")
            self.wallets[wallet.wallet_id] = wallet
    
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
            profit = trade_result.get('profit', 0.0)
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
            
            self.logger.info(f"✅ Evolution complete: {retirement_count} wallets retired and replaced")
            
        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
    
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
            
            return max(0.0, base_score + consistency_bonus - age_penalty)
            
        except Exception as e:
            self.logger.error(f"Error evaluating wallet: {e}")
            return 0.0
    
    async def _retire_wallet(self, wallet_id: str):
        """Retire a wallet"""
        try:
            wallet = self.wallets[wallet_id]
            wallet.state = WalletState.RETIRED
            self.retired_wallets.append(wallet_id)
            
            if wallet_id in self.active_wallets:
                self.active_wallets.remove(wallet_id)
            
            self.logger.info(f"🏁 Retired wallet {wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Error retiring wallet: {e}")
    
    async def _create_evolved_wallet(self, parent_wallet_id: str):
        """Create a new wallet evolved from a parent"""
        try:
            parent = self.wallets[parent_wallet_id]
            
            # Create new wallet ID
            new_wallet_id = f"evolved_{parent_wallet_id}_{int(datetime.now().timestamp())}"
            
            # Create evolved genetics
            evolved_genetics = self._crossover_genetics(parent.genetics)
            evolved_genetics = self._mutate_genetics(evolved_genetics)
            
            # Create new wallet
            wallet = await self._create_wallet(new_wallet_id)
            wallet.genetics = evolved_genetics
            wallet.behavior = parent.behavior  # Inherit behavior initially
            
            self.wallets[new_wallet_id] = wallet
            self.active_wallets.append(new_wallet_id)
            
            self.logger.info(f"🧬 Created evolved wallet {new_wallet_id} from {parent_wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating evolved wallet: {e}")
    
    def _crossover_genetics(self, parent_genetics: EvolutionGenetics) -> EvolutionGenetics:
        """Perform genetic crossover"""
        # Simple crossover - randomly inherit traits from parent
        return EvolutionGenetics(
            aggression=parent_genetics.aggression + secrets.SystemRandom().uniform(-0.1, 0.1),
            patience=parent_genetics.patience + secrets.SystemRandom().uniform(-0.1, 0.1),
            signal_trust=parent_genetics.signal_trust + secrets.SystemRandom().uniform(-0.1, 0.1),
            adaptation_rate=parent_genetics.adaptation_rate + secrets.SystemRandom().uniform(-0.1, 0.1),
            memory_strength=parent_genetics.memory_strength + secrets.SystemRandom().uniform(-0.1, 0.1),
            herd_immunity=parent_genetics.herd_immunity + secrets.SystemRandom().uniform(-0.1, 0.1)
        )
    
    def _mutate_genetics(self, genetics: EvolutionGenetics) -> EvolutionGenetics:
        """Apply random mutations to genetics"""
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
            
            # Update genetics based on performance
            await self._adapt_genetics(wallet)
            
            # Potentially change behavior
            if secrets.SystemRandom().random() < 0.1:  # 10% chance to change behavior
                wallet.behavior = secrets.SystemRandom().choice(list(WalletBehavior))
            
            wallet.last_evolution = datetime.now()
            wallet.evolution_count += 1
            
            self.logger.info(f"🧬 Evolved wallet {wallet_id} (evolution #{wallet.evolution_count})")
            
        except Exception as e:
            self.logger.error(f"Error evolving wallet: {e}")
    
    async def _adapt_genetics(self, wallet: TradingWallet):
        """Adapt genetics based on performance"""
        try:
            performance = wallet.performance
            
            # Adapt aggression based on win rate
            if performance.win_rate > 0.7:
                wallet.genetics.aggression = min(1.0, wallet.genetics.aggression + 0.05)
            elif performance.win_rate < 0.3:
                wallet.genetics.aggression = max(0.0, wallet.genetics.aggression - 0.05)
            
            # Adapt patience based on profit patterns
            if performance.avg_profit_per_trade > 0.1:
                wallet.genetics.patience = min(1.0, wallet.genetics.patience + 0.03)
            elif performance.avg_profit_per_trade < 0.01:
                wallet.genetics.patience = max(0.0, wallet.genetics.patience - 0.03)
            
            # Adapt signal trust based on recent performance
            if performance.consecutive_wins > 3:
                wallet.genetics.signal_trust = min(1.0, wallet.genetics.signal_trust + 0.02)
            elif performance.consecutive_losses > 3:
                wallet.genetics.signal_trust = max(0.0, wallet.genetics.signal_trust - 0.02)
            
        except Exception as e:
            self.logger.error(f"Error adapting genetics: {e}")
    
    async def _performance_tracking_loop(self):
        """Background performance tracking loop"""
        while self.initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Calculate Sharpe ratios and other metrics
                for wallet_id, wallet in self.wallets.items():
                    if wallet.state == WalletState.ACTIVE:
                        await self._update_advanced_metrics(wallet)
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    async def _update_advanced_metrics(self, wallet: TradingWallet):
        """Update advanced performance metrics"""
        try:
            performance = wallet.performance
            
            # Calculate Sharpe ratio (simplified)
            if performance.total_trades > 0:
                # Simplified Sharpe ratio calculation
                returns = [0.1 if i < performance.successful_trades else -0.05 for i in range(performance.total_trades)]
                if returns:
                    avg_return = sum(returns) / len(returns)
                    std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                    performance.sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            
            # Update max drawdown
            if performance.total_profit < 0:
                performance.max_drawdown = min(performance.max_drawdown, performance.total_profit)
            
        except Exception as e:
            self.logger.error(f"Error updating advanced metrics: {e}")
    
    async def _load_wallets_from_storage(self) -> bool:
        """Load wallets from persistent storage (JSON file)"""
        try:
            storage_path = Path('wallets/wallets.json')
            if not storage_path.exists():
                return False
            with open(storage_path, 'r') as f:
                data = json.load(f)
            for wallet_id, wallet_data in data.items():
                wallet = TradingWallet(**wallet_data)
                self.wallets[wallet_id] = wallet
            return True
        except Exception as e:
            self.logger.error(f"Error loading wallets: {e}")
            return False

    async def _save_wallets_to_storage(self):
        """Save wallets to persistent storage (JSON file)"""
        try:
            storage_path = Path('wallets/wallets.json')
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {wid: wallet.__dict__ for wid, wallet in self.wallets.items()}
            with open(storage_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving wallets: {e}")
    
    def get_wallet_status(self) -> Dict[str, Any]:
        """Get comprehensive wallet status"""
        try:
            active_count = len(self.active_wallets)
            retired_count = len(self.retired_wallets)
            total_count = len(self.wallets)
            
            avg_win_rate = 0.0
            avg_profit = 0.0
            
            if active_count > 0:
                win_rates = [self.wallets[wid].performance.win_rate for wid in self.active_wallets]
                profits = [self.wallets[wid].performance.total_profit for wid in self.active_wallets]
                avg_win_rate = sum(win_rates) / len(win_rates)
                avg_profit = sum(profits) / len(profits)
            
            return {
                'total_wallets': total_count,
                'active_wallets': active_count,
                'retired_wallets': retired_count,
                'avg_win_rate': avg_win_rate,
                'avg_total_profit': avg_profit,
                'evolution_active': self.evolution_active,
                'last_evolution': self.evolution_history[-1]['timestamp'] if self.evolution_history else None
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
            
            self.initialized = False
            self.logger.info("✅ Wallet manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global instance
_wallet_manager = None

async def get_wallet_manager() -> UnifiedWalletManager:
    """Get global wallet manager instance"""
    global _wallet_manager
    if _wallet_manager is None:
        _wallet_manager = UnifiedWalletManager()
        await _wallet_manager.initialize()
    return _wallet_manager 