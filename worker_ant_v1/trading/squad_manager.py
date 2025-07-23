"""
SQUAD MANAGER - ADHOCRATIC DYNAMIC TEAM FORMATION
================================================

The SquadManager enables the SwarmDecisionEngine to dynamically form
temporary, specialized "squads" of wallets to attack specific, high-value
market opportunities.

This implements the Adhocracy concept where teams are formed on-demand
based on opportunity characteristics and wallet genetics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from worker_ant_v1.core.wallet_manager import UnifiedWalletManager, TradingWallet, WalletBehavior
from worker_ant_v1.utils.logger import get_logger


class SquadType(Enum):
    """Types of specialized squads"""
    SNIPER = "sniper"          # For new launches and quick entries
    WHALE_WATCH = "whale_watch"  # To mimic smart money movements
    SCALPER = "scalper"        # For high volatility and quick profits
    ACCUMULATOR = "accumulator"  # For building positions over time
    FOMO = "fomo"              # For trending and viral tokens
    STEALTH = "stealth"        # For low-key, under-the-radar trades
    LIQUIDITY_PROVISION = "liquidity_provision"  # For adding/monitoring liquidity with rug-pull immunity


@dataclass
class SquadRuleset:
    """Temporary ruleset for squad operations"""
    max_position_size_sol: float = 50.0
    risk_level: str = "high"
    slippage_tolerance: float = 0.05  # 5% slippage
    hold_duration_minutes: int = 30
    entry_speed: str = "immediate"  # immediate, gradual, stealth
    exit_strategy: str = "aggressive"  # aggressive, conservative, trailing
    coordination_mode: str = "synchronized"  # synchronized, staggered, random


@dataclass
class Squad:
    """Dynamic squad for specific market opportunities"""
    squad_id: str
    squad_type: SquadType
    mission_target_token: str
    wallet_ids: List[str]
    ruleset: SquadRuleset
    created_at: datetime
    mission_duration_minutes: int = 60
    status: str = "active"  # active, completed, disbanded, failed
    mission_results: Dict[str, Any] = field(default_factory=dict)
    disbanded_at: Optional[datetime] = None


class SquadManager:
    """Manages dynamic squad formation and operations"""
    
    def __init__(self):
        self.logger = get_logger("SquadManager")
        
        # Core systems
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        
        # Squad management
        self.active_squads: Dict[str, Squad] = {}
        self.squad_history: List[Squad] = []
        self.squad_counter = 0
        
        # Squad type configurations
        self.squad_configs = {
            SquadType.SNIPER: {
                'required_wallets': 3,
                'preferred_behaviors': [WalletBehavior.SNIPER, WalletBehavior.SCALPER],
                'min_aggression': 0.7,
                'max_patience': 0.3,
                'ruleset': SquadRuleset(
                    max_position_size_sol=30.0,
                    risk_level="high",
                    slippage_tolerance=0.08,
                    hold_duration_minutes=15,
                    entry_speed="immediate",
                    exit_strategy="aggressive"
                )
            },
            SquadType.WHALE_WATCH: {
                'required_wallets': 2,
                'preferred_behaviors': [WalletBehavior.MIMICKER, WalletBehavior.ACCUMULATOR],
                'min_aggression': 0.5,
                'max_patience': 0.8,
                'ruleset': SquadRuleset(
                    max_position_size_sol=40.0,
                    risk_level="medium",
                    slippage_tolerance=0.03,
                    hold_duration_minutes=120,
                    entry_speed="gradual",
                    exit_strategy="conservative"
                )
            },
            SquadType.SCALPER: {
                'required_wallets': 4,
                'preferred_behaviors': [WalletBehavior.SCALPER, WalletBehavior.SNIPER],
                'min_aggression': 0.6,
                'max_patience': 0.4,
                'ruleset': SquadRuleset(
                    max_position_size_sol=25.0,
                    risk_level="medium",
                    slippage_tolerance=0.04,
                    hold_duration_minutes=45,
                    entry_speed="immediate",
                    exit_strategy="aggressive"
                )
            },
            SquadType.ACCUMULATOR: {
                'required_wallets': 2,
                'preferred_behaviors': [WalletBehavior.ACCUMULATOR, WalletBehavior.HODLER],
                'min_aggression': 0.3,
                'max_patience': 0.9,
                'ruleset': SquadRuleset(
                    max_position_size_sol=60.0,
                    risk_level="low",
                    slippage_tolerance=0.02,
                    hold_duration_minutes=480,  # 8 hours
                    entry_speed="gradual",
                    exit_strategy="conservative"
                )
            },
            SquadType.FOMO: {
                'required_wallets': 5,
                'preferred_behaviors': [WalletBehavior.SNIPER, WalletBehavior.SCALPER],
                'min_aggression': 0.8,
                'max_patience': 0.2,
                'ruleset': SquadRuleset(
                    max_position_size_sol=35.0,
                    risk_level="high",
                    slippage_tolerance=0.06,
                    hold_duration_minutes=30,
                    entry_speed="immediate",
                    exit_strategy="aggressive"
                )
            },
            SquadType.STEALTH: {
                'required_wallets': 2,
                'preferred_behaviors': [WalletBehavior.SCALPER, WalletBehavior.ACCUMULATOR],
                'min_aggression': 0.4,
                'max_patience': 0.6,
                'ruleset': SquadRuleset(
                    max_position_size_sol=20.0,
                    risk_level="medium",
                    slippage_tolerance=0.03,
                    hold_duration_minutes=90,
                    entry_speed="stealth",
                    exit_strategy="conservative"
                )
            },
            SquadType.LIQUIDITY_PROVISION: {
                'required_wallets': 3,
                'preferred_behaviors': [WalletBehavior.ACCUMULATOR, WalletBehavior.MIMICKER],
                'min_aggression': 0.6,
                'max_patience': 0.9,
                'ruleset': SquadRuleset(
                    max_position_size_sol=80.0,  # Larger amounts for liquidity provision
                    risk_level="high",
                    slippage_tolerance=0.02,
                    hold_duration_minutes=360,  # 6 hours monitoring
                    entry_speed="gradual",
                    exit_strategy="immediate",  # Immediate exit on rug detection
                    coordination_mode="synchronized"
                )
            }
        }
        
        # System state
        self.initialized = False
        
        self.logger.info("🎯 Squad Manager initialized")
    
    async def initialize(self, wallet_manager: UnifiedWalletManager) -> bool:
        """Initialize the squad manager"""
        try:
            self.logger.info("🎯 Initializing Squad Manager...")
            
            self.wallet_manager = wallet_manager
            
            # Start background tasks
            asyncio.create_task(self._squad_monitoring_loop())
            asyncio.create_task(self._cleanup_expired_squads())
            
            self.initialized = True
            self.logger.info("✅ Squad Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Squad Manager initialization failed: {e}")
            return False
    
    async def form_squad_for_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Squad]:
        """Form a squad for a specific market opportunity"""
        try:
            token_address = opportunity.get('token_address', '')
            token_age_hours = opportunity.get('token_age_hours', 24)
            market_cap = opportunity.get('market_cap', 0)
            volume_24h = opportunity.get('volume_24h', 0)
            is_trending = opportunity.get('is_trending', False)
            volatility = opportunity.get('volatility', 0.5)
            
            # Determine squad type based on opportunity characteristics
            squad_type = self._determine_squad_type(
                token_age_hours, market_cap, volume_24h, is_trending, volatility
            )
            
            if not squad_type:
                self.logger.info(f"📋 No special squad needed for {token_address}")
                return None
            
            # Select suitable wallets for the squad
            selected_wallets = await self._select_squad_wallets(squad_type)
            
            if not selected_wallets:
                self.logger.warning(f"⚠️ No suitable wallets found for {squad_type.value} squad")
                return None
            
            # Create squad
            squad = await self._create_squad(squad_type, token_address, selected_wallets)
            
            # Apply squad ruleset to selected wallets
            await self._apply_squad_ruleset(squad)
            
            self.logger.info(f"🎯 Formed {squad_type.value} squad for {token_address}")
            return squad
            
        except Exception as e:
            self.logger.error(f"❌ Error forming squad: {e}")
            return None
    
    def _determine_squad_type(self, token_age_hours: float, market_cap: float, 
                             volume_24h: float, is_trending: bool, volatility: float) -> Optional[SquadType]:
        """Determine the appropriate squad type for an opportunity"""
        try:
            # LIQUIDITY_PROVISION: High-conviction trades needing liquidity support and rug protection
            liquidity_provision_criteria = (
                hasattr(self, '_should_use_liquidity_provision') and 
                self._should_use_liquidity_provision(token_age_hours, market_cap, volume_24h, volatility)
            )
            if liquidity_provision_criteria:
                return SquadType.LIQUIDITY_PROVISION
            
            # SNIPER: Very new tokens (< 1 hour)
            if token_age_hours < 1:
                return SquadType.SNIPER
            
            # FOMO: Trending tokens with high volume
            if is_trending and volume_24h > 1000000:  # > $1M volume
                return SquadType.FOMO
            
            # SCALPER: High volatility tokens
            if volatility > 0.8:
                return SquadType.SCALPER
            
            # WHALE_WATCH: Large market cap tokens
            if market_cap > 10000000:  # > $10M market cap
                return SquadType.WHALE_WATCH
            
            # ACCUMULATOR: Stable, established tokens
            if token_age_hours > 24 and volatility < 0.3:
                return SquadType.ACCUMULATOR
            
            # STEALTH: Medium age, medium volatility
            if 1 <= token_age_hours <= 24 and 0.3 <= volatility <= 0.7:
                return SquadType.STEALTH
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining squad type: {e}")
            return None
    
    async def _select_squad_wallets(self, squad_type: SquadType) -> List[str]:
        """Select suitable wallets for a squad based on genetics and behavior"""
        try:
            if not self.wallet_manager:
                return []
            
            config = self.squad_configs[squad_type]
            required_count = config['required_wallets']
            preferred_behaviors = config['preferred_behaviors']
            min_aggression = config['min_aggression']
            max_patience = config['max_patience']
            
            # Get all active wallets
            active_wallets = []
            for wallet_id in self.wallet_manager.active_wallets:
                wallet = self.wallet_manager.wallets.get(wallet_id)
                if wallet and wallet.state.value == "active":
                    # Check if wallet is already in a squad
                    if not self._is_wallet_in_squad(wallet_id):
                        active_wallets.append(wallet)
            
            # Score wallets based on squad requirements
            wallet_scores = []
            for wallet in active_wallets:
                score = self._calculate_wallet_squad_fitness(wallet, squad_type, config)
                wallet_scores.append((wallet.wallet_id, score))
            
            # Sort by fitness score
            wallet_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top wallets
            selected_wallets = []
            for wallet_id, score in wallet_scores[:required_count]:
                if score > 0.5:  # Minimum fitness threshold
                    selected_wallets.append(wallet_id)
            
            return selected_wallets
            
        except Exception as e:
            self.logger.error(f"Error selecting squad wallets: {e}")
            return []
    
    def _calculate_wallet_squad_fitness(self, wallet: TradingWallet, squad_type: SquadType, 
                                       config: Dict[str, Any]) -> float:
        """Calculate how well a wallet fits a squad type"""
        try:
            score = 0.0
            
            # Behavior match (40% weight)
            if wallet.behavior in config['preferred_behaviors']:
                score += 0.4
            
            # Genetics match (60% weight)
            genetics = wallet.genetics
            
            # Aggression check
            if genetics.aggression >= config['min_aggression']:
                score += 0.2
            
            # Patience check
            if genetics.patience <= config['max_patience']:
                score += 0.2
            
            # Signal trust (higher is better for squads)
            score += genetics.signal_trust * 0.1
            
            # Adaptation rate (higher is better for dynamic squads)
            score += genetics.adaptation_rate * 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating wallet fitness: {e}")
            return 0.0
    
    def _is_wallet_in_squad(self, wallet_id: str) -> bool:
        """Check if a wallet is already part of an active squad"""
        for squad in self.active_squads.values():
            if wallet_id in squad.wallet_ids:
                return True
        return False
    
    async def _create_squad(self, squad_type: SquadType, token_address: str, 
                           wallet_ids: List[str]) -> Squad:
        """Create a new squad"""
        try:
            self.squad_counter += 1
            squad_id = f"squad_{squad_type.value}_{self.squad_counter:03d}"
            
            config = self.squad_configs[squad_type]
            
            squad = Squad(
                squad_id=squad_id,
                squad_type=squad_type,
                mission_target_token=token_address,
                wallet_ids=wallet_ids,
                ruleset=config['ruleset'],
                created_at=datetime.now(),
                mission_duration_minutes=config['ruleset'].hold_duration_minutes * 2
            )
            
            self.active_squads[squad_id] = squad
            
            self.logger.info(f"🎯 Created squad {squad_id} with {len(wallet_ids)} wallets")
            return squad
            
        except Exception as e:
            self.logger.error(f"Error creating squad: {e}")
            raise
    
    async def _apply_squad_ruleset(self, squad: Squad):
        """Apply squad ruleset to selected wallets"""
        try:
            for wallet_id in squad.wallet_ids:
                wallet = self.wallet_manager.wallets.get(wallet_id)
                if wallet:
                    # Set active squad ruleset
                    wallet.active_squad_ruleset = {
                        'squad_id': squad.squad_id,
                        'max_position_size_sol': squad.ruleset.max_position_size_sol,
                        'risk_level': squad.ruleset.risk_level,
                        'slippage_tolerance': squad.ruleset.slippage_tolerance,
                        'hold_duration_minutes': squad.ruleset.hold_duration_minutes,
                        'entry_speed': squad.ruleset.entry_speed,
                        'exit_strategy': squad.ruleset.exit_strategy,
                        'coordination_mode': squad.ruleset.coordination_mode
                    }
                    
                    self.logger.info(f"⚙️ Applied squad ruleset to wallet {wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Error applying squad ruleset: {e}")
    
    async def disband_squad(self, squad_id: str, reason: str = "mission_complete"):
        """Disband a squad and reset wallet rulesets"""
        try:
            squad = self.active_squads.get(squad_id)
            if not squad:
                return False
            
            # Reset wallet rulesets
            for wallet_id in squad.wallet_ids:
                wallet = self.wallet_manager.wallets.get(wallet_id)
                if wallet:
                    wallet.active_squad_ruleset = None
                    self.logger.info(f"🔄 Reset ruleset for wallet {wallet_id}")
            
            # Update squad status
            squad.status = "disbanded"
            squad.disbanded_at = datetime.now()
            squad.mission_results['disband_reason'] = reason
            
            # Move to history
            self.squad_history.append(squad)
            del self.active_squads[squad_id]
            
            self.logger.info(f"🔄 Disbanded squad {squad_id}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disbanding squad {squad_id}: {e}")
            return False
    
    async def get_squad_status(self, squad_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific squad"""
        try:
            squad = self.active_squads.get(squad_id)
            if not squad:
                return None
            
            return {
                'squad_id': squad.squad_id,
                'squad_type': squad.squad_type.value,
                'mission_target_token': squad.mission_target_token,
                'wallet_count': len(squad.wallet_ids),
                'status': squad.status,
                'created_at': squad.created_at.isoformat(),
                'mission_duration_minutes': squad.mission_duration_minutes,
                'ruleset': {
                    'max_position_size_sol': squad.ruleset.max_position_size_sol,
                    'risk_level': squad.ruleset.risk_level,
                    'entry_speed': squad.ruleset.entry_speed,
                    'exit_strategy': squad.ruleset.exit_strategy
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting squad status: {e}")
            return None
    
    async def _squad_monitoring_loop(self):
        """Monitor active squads"""
        while self.initialized:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for squad_id, squad in list(self.active_squads.items()):
                    # Check if squad has exceeded mission duration
                    mission_end = squad.created_at + timedelta(minutes=squad.mission_duration_minutes)
                    if datetime.now() > mission_end:
                        await self.disband_squad(squad_id, "mission_timeout")
                
            except Exception as e:
                self.logger.error(f"Squad monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_squads(self):
        """Clean up old squad history"""
        while self.initialized:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Keep only last 100 squads in history
                if len(self.squad_history) > 100:
                    self.squad_history = self.squad_history[-100:]
                
            except Exception as e:
                self.logger.error(f"Squad cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def _should_use_liquidity_provision(self, token_age_hours: float, market_cap: float,
                                       volume_24h: float, volatility: float) -> bool:
        """Determine if liquidity provision squad should be deployed"""
        try:
            
            # 1. High-conviction trade (would be passed as additional parameter in real implementation)
            liquidity_provision_criteria = [
                token_age_hours <= 12,           # New enough to influence
                market_cap < 5000000,            # Small enough that our liquidity matters  
                volume_24h > 100000,             # Sufficient volume to warrant attention
                0.3 <= volatility <= 0.9         # Manageable volatility range
            ]
            
                market_cap < 5000000,            # Small enough that our liquidity matters  
            return sum(liquidity_provision_criteria) >= 3
            
        except Exception as e:
            self.logger.error(f"Liquidity provision criteria error: {e}")
            return False

    def get_squad_manager_status(self) -> Dict[str, Any]:
        """Get overall squad manager status"""
        try:
            return {
                'active_squads': len(self.active_squads),
                'total_squads_formed': self.squad_counter,
                'squad_history_size': len(self.squad_history),
                'squad_types': {
                    squad_type.value: len([s for s in self.active_squads.values() if s.squad_type == squad_type])
                    for squad_type in SquadType
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting squad manager status: {e}")
            return {}


async def get_squad_manager() -> SquadManager:
    """Get global squad manager instance"""
    from worker_ant_v1.core.wallet_manager import get_wallet_manager
    
    wallet_manager = await get_wallet_manager()
    squad_manager = SquadManager()
    await squad_manager.initialize(wallet_manager)
    return squad_manager 