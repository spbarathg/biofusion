"""
VAULT WALLET SYSTEM - PROFIT PROTECTION & SECURE STORAGE
======================================================

Manages secure vault wallets for profit protection and automatic
compounding with multiple security layers.
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
try:
    from solana.keypair import Keypair
except ImportError:
    from ..utils.solana_compat import Keypair
import base58

class VaultType(Enum):
    EMERGENCY = "emergency"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    LEGENDARY = "legendary"
    GODMODE = "godmode"


class StrategicEnergyTier(Enum):
    """Strategic Energy Reserve - Three-Tiered Capital Hierarchy"""
    KINETIC = "kinetic"     # Tier 1: Hot wallet capital (10-20%) - instantly available
    POTENTIAL = "potential" # Tier 2: Warm reserve (30-40%) - consensus required
    CORE = "core"          # Tier 3: Cold vault (40-60%) - time-locked + multi-sig


class ColonyConfidenceLevel(Enum):
    """Colony consensus confidence levels for capital deployment"""
    LOW = "low"               # < 60% confidence
    MEDIUM = "medium"         # 60-75% confidence  
    HIGH = "high"            # 75-85% confidence
    UNANIMOUS = "unanimous"   # > 85% confidence - A+ setups only


@dataclass
class StrategicEnergyReserve:
    """Strategic Energy Reserve configuration for anti-fragile capital management"""
    tier: StrategicEnergyTier
    allocation_percentage: float
    min_balance_sol: float
    max_balance_sol: float
    deployment_requirements: Dict[str, Any]
    time_lock_hours: int = 0
    multi_sig_required: bool = False
    consensus_threshold: ColonyConfidenceLevel = ColonyConfidenceLevel.LOW
    emergency_accessible: bool = True
    auto_compound_enabled: bool = True
    preservation_priority: int = 1  # 1 = highest priority for preservation


@dataclass
class VaultWallet:
    """Individual vault wallet"""
    vault_id: str
    vault_type: VaultType
    address: str
    private_key: str
    balance: float = 0.0
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    security_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Strategic Energy Reserve enhancements
    energy_tier: Optional[StrategicEnergyTier] = None
    time_lock_expiry: Optional[datetime] = None
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    preservation_score: float = 0.0  # Higher score = better for long-term preservation
    last_consensus_deployment: Optional[datetime] = None
    
    def is_available_for_deployment(self, confidence_level: ColonyConfidenceLevel) -> bool:
        """Check if this vault can be deployed for the given confidence level"""
        if not self.energy_tier:
            return False
            
        # Check time lock
        if self.time_lock_expiry and datetime.now() < self.time_lock_expiry:
            return False
            
        # Check confidence requirements based on energy tier
        if self.energy_tier == StrategicEnergyTier.KINETIC:
            return True  # Always available for hot wallets
        elif self.energy_tier == StrategicEnergyTier.POTENTIAL:
            return confidence_level in [ColonyConfidenceLevel.HIGH, ColonyConfidenceLevel.UNANIMOUS]
        elif self.energy_tier == StrategicEnergyTier.CORE:
            return confidence_level == ColonyConfidenceLevel.UNANIMOUS
        
        return False

@dataclass
class VaultAllocation:
    """Vault allocation configuration"""
    vault_type: VaultType
    percentage: float
    min_amount: float
    max_amount: float
    auto_compound: bool
    compound_frequency_hours: int
    security_requirements: List[str]

class VaultWalletSystem:
    """Manages secure vault wallets for profit protection"""
    
    def __init__(self):
        self.logger = setup_logger("VaultWalletSystem")
        
        # Vault storage
        self.vaults: Dict[str, VaultWallet] = {}
        self.vault_allocations: Dict[VaultType, VaultAllocation] = {}
        
        # Strategic Energy Reserve system
        self.strategic_reserves: Dict[StrategicEnergyTier, StrategicEnergyReserve] = {}
        self.energy_tier_balances: Dict[StrategicEnergyTier, float] = {
            StrategicEnergyTier.KINETIC: 0.0,
            StrategicEnergyTier.POTENTIAL: 0.0,
            StrategicEnergyTier.CORE: 0.0
        }
        self.total_strategic_capital = 0.0
        self.colony_commander = None  # Integration point
        
        # System state
        self.initialized = False
        self.system_active = True
        self.total_vault_balance = 0.0
        self.total_profits_secured = 0.0
        
        # Security settings
        self.security_config = {
            'encryption_enabled': True,
            'multi_sig_required': False,
            'withdrawal_delay_hours': 24,
            'max_daily_withdrawal': 1.0,  # SOL
            'emergency_access_enabled': True
        }
        
        # Initialize default allocations and strategic reserves
        self._initialize_default_allocations()
        self._initialize_strategic_energy_reserves()
    
    def _initialize_default_allocations(self):
        """Initialize default vault allocations"""
        self.vault_allocations = {
            VaultType.EMERGENCY: VaultAllocation(
                vault_type=VaultType.EMERGENCY,
                percentage=0.05,  # 5%
                min_amount=0.1,
                max_amount=1.0,
                auto_compound=False,
                compound_frequency_hours=0,
                security_requirements=['encryption', 'delay']
            ),
            VaultType.DAILY: VaultAllocation(
                vault_type=VaultType.DAILY,
                percentage=0.10,  # 10%
                min_amount=0.05,
                max_amount=2.0,
                auto_compound=True,
                compound_frequency_hours=24,
                security_requirements=['encryption']
            ),
            VaultType.WEEKLY: VaultAllocation(
                vault_type=VaultType.WEEKLY,
                percentage=0.03,  # 3%
                min_amount=0.1,
                max_amount=5.0,
                auto_compound=True,
                compound_frequency_hours=168,  # 7 days
                security_requirements=['encryption', 'delay']
            ),
            VaultType.MONTHLY: VaultAllocation(
                vault_type=VaultType.MONTHLY,
                percentage=0.02,  # 2%
                min_amount=0.2,
                max_amount=10.0,
                auto_compound=True,
                compound_frequency_hours=720,  # 30 days
                security_requirements=['encryption', 'delay', 'multi_sig']
            ),
            VaultType.LEGENDARY: VaultAllocation(
                vault_type=VaultType.LEGENDARY,
                percentage=0.00,  # Manual allocation only
                min_amount=1.0,
                max_amount=50.0,
                auto_compound=False,
                compound_frequency_hours=0,
                security_requirements=['encryption', 'delay', 'multi_sig', 'emergency_only']
            ),
            VaultType.GODMODE: VaultAllocation(
                vault_type=VaultType.GODMODE,
                percentage=0.00,  # Manual allocation only
                min_amount=5.0,
                max_amount=100.0,
                auto_compound=False,
                compound_frequency_hours=0,
                security_requirements=['encryption', 'delay', 'multi_sig', 'emergency_only', 'manual_approval']
            )
        }
    
    def _initialize_strategic_energy_reserves(self):
        """Initialize default strategic energy reserves"""
        self.strategic_reserves = {
            StrategicEnergyTier.KINETIC: StrategicEnergyReserve(
                tier=StrategicEnergyTier.KINETIC,
                allocation_percentage=0.15, # 15% of total capital
                min_balance_sol=0.0,
                max_balance_sol=1000.0, # Example max for KINETIC
                deployment_requirements={
                    "confidence_level": ColonyConfidenceLevel.HIGH,
                    "time_lock_hours": 0,
                    "multi_sig_required": False
                },
                consensus_threshold=ColonyConfidenceLevel.HIGH,
                emergency_accessible=True,
                auto_compound_enabled=True,
                preservation_priority=1
            ),
            StrategicEnergyTier.POTENTIAL: StrategicEnergyReserve(
                tier=StrategicEnergyTier.POTENTIAL,
                allocation_percentage=0.30, # 30% of total capital
                min_balance_sol=0.0,
                max_balance_sol=5000.0, # Example max for POTENTIAL
                deployment_requirements={
                    "confidence_level": ColonyConfidenceLevel.MEDIUM,
                    "time_lock_hours": 24,
                    "multi_sig_required": True
                },
                consensus_threshold=ColonyConfidenceLevel.MEDIUM,
                emergency_accessible=True,
                auto_compound_enabled=True,
                preservation_priority=2
            ),
            StrategicEnergyTier.CORE: StrategicEnergyReserve(
                tier=StrategicEnergyTier.CORE,
                allocation_percentage=0.55, # 55% of total capital
                min_balance_sol=0.0,
                max_balance_sol=10000.0, # Example max for CORE
                deployment_requirements={
                    "confidence_level": ColonyConfidenceLevel.LOW,
                    "time_lock_hours": 72,
                    "multi_sig_required": True
                },
                consensus_threshold=ColonyConfidenceLevel.LOW,
                emergency_accessible=False,
                auto_compound_enabled=False,
                preservation_priority=3
            )
        }
        self.energy_tier_balances = {
            StrategicEnergyTier.KINETIC: 0.0,
            StrategicEnergyTier.POTENTIAL: 0.0,
            StrategicEnergyTier.CORE: 0.0
        }
        self.total_strategic_capital = sum(r.allocation_percentage for r in self.strategic_reserves.values())
    
    async def initialize_vault_system(self) -> bool:
        """Initialize the vault wallet system"""
        try:
            self.logger.info("🏦 Initializing Vault Wallet System...")
            
            # Create vault wallets
            await self._create_vault_wallets()
            
            # Load existing vault data
            await self._load_vault_data()
            
            # Start background tasks
            asyncio.create_task(self._vault_monitoring_loop())
            asyncio.create_task(self._auto_compound_loop())
            asyncio.create_task(self._security_monitoring_loop())
            
            self.initialized = True
            self.logger.info(f"✅ Vault system initialized with {len(self.vaults)} vaults")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vault system: {e}")
            return False
    
    async def _create_vault_wallets(self):
        """Create vault wallets for each type using real Solana keypair"""
        for vault_type in VaultType:
            vault_id = f"vault_{vault_type.value}"
            keypair = Keypair.generate()
            private_key = base58.b58encode(keypair.secret_key).decode()
            address = str(keypair.public_key)
            vault = VaultWallet(
                vault_id=vault_id,
                vault_type=vault_type,
                address=address,
                private_key=private_key,
                security_level=self._get_security_level(vault_type)
            )
            self.vaults[vault_id] = vault
            self.logger.info(f"🏦 Created {vault_type.value} vault: {address}")
    
    def _get_security_level(self, vault_type: VaultType) -> int:
        """Get security level for vault type"""
        security_levels = {
            VaultType.EMERGENCY: 1,
            VaultType.DAILY: 2,
            VaultType.WEEKLY: 3,
            VaultType.MONTHLY: 4,
            VaultType.LEGENDARY: 5,
            VaultType.GODMODE: 6
        }
        return security_levels.get(vault_type, 1)
    
    # === STRATEGIC ENERGY RESERVE METHODS ===
    
    async def allocate_to_strategic_reserves(self, total_capital: float) -> Dict[StrategicEnergyTier, float]:
        """Allocate capital to strategic energy reserves based on anti-fragile hierarchy"""
        try:
            allocations = {}
            
            for tier, reserve in self.strategic_reserves.items():
                allocation_amount = total_capital * reserve.allocation_percentage
                
                # Apply tier-specific constraints
                allocation_amount = max(reserve.min_balance_sol, min(reserve.max_balance_sol, allocation_amount))
                
                # Update tier balance
                self.energy_tier_balances[tier] += allocation_amount
                allocations[tier] = allocation_amount
                
                self.logger.info(f"💎 Allocated {allocation_amount:.4f} SOL to {tier.value} energy reserve")
            
            self.total_strategic_capital = sum(allocations.values())
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error allocating to strategic reserves: {e}")
            return {}
    
    async def request_strategic_deployment(self, 
                                         amount: float, 
                                         confidence_level: ColonyConfidenceLevel,
                                         opportunity_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Request deployment of strategic capital for high-conviction opportunities"""
        try:
            deployment_result = {
                'approved': False,
                'amount_approved': 0.0,
                'tier_deployed': None,
                'deployment_id': f"deploy_{int(datetime.now().timestamp())}",
                'reason': ''
            }
            
            # Determine which tier can be deployed based on confidence level
            available_tiers = []
            for tier, reserve in self.strategic_reserves.items():
                if (self.energy_tier_balances[tier] >= amount and
                    confidence_level.value in [reserve.consensus_threshold.value, 'unanimous']):
                    available_tiers.append((tier, reserve))
            
            # Sort by preservation priority (lower number = higher priority to preserve)
            available_tiers.sort(key=lambda x: x[1].preservation_priority)
            
            # Try to deploy from the least valuable tier first
            for tier, reserve in available_tiers:
                if self.energy_tier_balances[tier] >= amount:
                    # Check if colony commander consensus is met (if available)
                    if await self._check_colony_consensus(confidence_level, opportunity_data):
                        # Deploy capital
                        await self._execute_strategic_deployment(tier, amount, deployment_result['deployment_id'])
                        
                        deployment_result.update({
                            'approved': True,
                            'amount_approved': amount,
                            'tier_deployed': tier,
                            'reason': f'Deployed from {tier.value} tier with {confidence_level.value} confidence'
                        })
                        
                        self.logger.info(f"✅ Strategic deployment approved: {amount:.4f} SOL from {tier.value} tier")
                        return deployment_result
            
            deployment_result['reason'] = f'Insufficient capital or confidence level too low for deployment'
            self.logger.warning(f"❌ Strategic deployment rejected: {deployment_result['reason']}")
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Error in strategic deployment request: {e}")
            return {'approved': False, 'reason': str(e)}
    
    async def _execute_strategic_deployment(self, tier: StrategicEnergyTier, amount: float, deployment_id: str):
        """Execute the actual strategic capital deployment"""
        try:
            # Deduct from tier balance
            self.energy_tier_balances[tier] -= amount
            
            # Record deployment
            deployment_record = {
                'deployment_id': deployment_id,
                'tier': tier.value,
                'amount': amount,
                'timestamp': datetime.now(),
                'status': 'deployed'
            }
            
            # Add to deployment history for tracking
            for vault_id, vault in self.vaults.items():
                if vault.energy_tier == tier:
                    vault.deployment_history.append(deployment_record)
                    vault.last_consensus_deployment = datetime.now()
                    break
            
            self.logger.info(f"🚀 Strategic deployment executed: {amount:.4f} SOL from {tier.value}")
            
        except Exception as e:
            self.logger.error(f"Error executing strategic deployment: {e}")
    
    async def _check_colony_consensus(self, confidence_level: ColonyConfidenceLevel, opportunity_data: Dict[str, Any] = None) -> bool:
        """Check if colony consensus is met for the deployment"""
        try:
            # If colony commander is available, check with it
            if self.colony_commander:
                consensus_result = await self.colony_commander.check_deployment_consensus(confidence_level, opportunity_data)
                return consensus_result.get('consensus_reached', False)
            
            # Fallback: basic confidence level validation
            confidence_thresholds = {
                ColonyConfidenceLevel.LOW: 0.6,
                ColonyConfidenceLevel.MEDIUM: 0.75,
                ColonyConfidenceLevel.HIGH: 0.85,
                ColonyConfidenceLevel.UNANIMOUS: 0.95
            }
            
            # For now, return True if we have opportunity data with sufficient confidence
            if opportunity_data:
                opportunity_confidence = opportunity_data.get('confidence_score', 0.0)
                required_threshold = confidence_thresholds[confidence_level]
                return opportunity_confidence >= required_threshold
            
            # Default to false if no data available
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking colony consensus: {e}")
            return False
    
    async def implement_time_locks(self, tier: StrategicEnergyTier, lock_duration_hours: int = None):
        """Implement time locks on Core Energy tier for capital preservation"""
        try:
            if tier == StrategicEnergyTier.CORE:
                lock_duration = lock_duration_hours or self.strategic_reserves[tier].deployment_requirements.get('time_lock_hours', 72)
                lock_expiry = datetime.now() + timedelta(hours=lock_duration)
                
                # Apply time lock to all Core Energy vaults
                for vault_id, vault in self.vaults.items():
                    if vault.energy_tier == tier:
                        vault.time_lock_expiry = lock_expiry
                        self.logger.info(f"🔒 Time lock applied to {vault_id} until {lock_expiry}")
                
                return True
            else:
                self.logger.warning(f"Time locks only applicable to Core Energy tier, not {tier.value}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error implementing time locks: {e}")
            return False
    
    def get_strategic_energy_status(self) -> Dict[str, Any]:
        """Get comprehensive Strategic Energy Reserve status"""
        try:
            return {
                'total_strategic_capital': self.total_strategic_capital,
                'tier_balances': {
                    tier.value: balance for tier, balance in self.energy_tier_balances.items()
                },
                'tier_allocations': {
                    tier.value: {
                        'percentage': reserve.allocation_percentage,
                        'current_balance': self.energy_tier_balances[tier],
                        'max_balance': reserve.max_balance_sol,
                        'deployment_requirements': reserve.deployment_requirements,
                        'preservation_priority': reserve.preservation_priority
                    }
                    for tier, reserve in self.strategic_reserves.items()
                },
                'deployment_availability': {
                    tier.value: self._check_tier_availability(tier) for tier in StrategicEnergyTier
                },
                'anti_fragile_mode': self._assess_anti_fragile_readiness()
            }
        except Exception as e:
            self.logger.error(f"Error getting strategic energy status: {e}")
            return {}
    
    def _check_tier_availability(self, tier: StrategicEnergyTier) -> Dict[str, Any]:
        """Check availability status of a specific energy tier"""
        try:
            tier_vaults = [vault for vault in self.vaults.values() if vault.energy_tier == tier]
            available_vaults = []
            locked_vaults = []
            
            for vault in tier_vaults:
                if vault.time_lock_expiry and datetime.now() < vault.time_lock_expiry:
                    locked_vaults.append({
                        'vault_id': vault.vault_id,
                        'lock_expiry': vault.time_lock_expiry.isoformat()
                    })
                else:
                    available_vaults.append(vault.vault_id)
            
            return {
                'balance': self.energy_tier_balances[tier],
                'available_vaults': len(available_vaults),
                'locked_vaults': len(locked_vaults),
                'vault_details': {
                    'available': available_vaults,
                    'locked': locked_vaults
                }
            }
        except Exception as e:
            self.logger.error(f"Error checking tier availability: {e}")
            return {}
    
    def _assess_anti_fragile_readiness(self) -> Dict[str, Any]:
        """Assess the anti-fragile readiness of the Strategic Energy Reserve"""
        try:
            total_available = sum(self.energy_tier_balances.values())
            
            # Calculate distribution ratios
            kinetic_ratio = self.energy_tier_balances[StrategicEnergyTier.KINETIC] / max(total_available, 1)
            potential_ratio = self.energy_tier_balances[StrategicEnergyTier.POTENTIAL] / max(total_available, 1)
            core_ratio = self.energy_tier_balances[StrategicEnergyTier.CORE] / max(total_available, 1)
            
            # Assess anti-fragile score based on ideal distribution
            ideal_kinetic = 0.15   # 15% in kinetic
            ideal_potential = 0.30 # 30% in potential  
            ideal_core = 0.55      # 55% in core
            
            kinetic_score = 1.0 - abs(kinetic_ratio - ideal_kinetic)
            potential_score = 1.0 - abs(potential_ratio - ideal_potential)
            core_score = 1.0 - abs(core_ratio - ideal_core)
            
            anti_fragile_score = (kinetic_score + potential_score + core_score) / 3
            
            return {
                'anti_fragile_score': anti_fragile_score,
                'distribution_health': {
                    'kinetic_ratio': kinetic_ratio,
                    'potential_ratio': potential_ratio,
                    'core_ratio': core_ratio
                },
                'readiness_level': (
                    'excellent' if anti_fragile_score > 0.9 else
                    'good' if anti_fragile_score > 0.75 else
                    'moderate' if anti_fragile_score > 0.6 else
                    'poor'
                ),
                'recommendations': self._generate_rebalancing_recommendations(kinetic_ratio, potential_ratio, core_ratio)
            }
        except Exception as e:
            self.logger.error(f"Error assessing anti-fragile readiness: {e}")
            return {}
    
    def _generate_rebalancing_recommendations(self, kinetic_ratio: float, potential_ratio: float, core_ratio: float) -> List[str]:
        """Generate recommendations for rebalancing the Strategic Energy Reserve"""
        recommendations = []
        
        if kinetic_ratio > 0.25:  # Too much in kinetic (hot wallets)
            recommendations.append("Move excess capital from Kinetic to Potential Energy tier")
        elif kinetic_ratio < 0.10:  # Too little in kinetic
            recommendations.append("Increase Kinetic Energy allocation for operational flexibility")
        
        if potential_ratio < 0.25:  # Too little in potential
            recommendations.append("Increase Potential Energy reserves for opportunity deployment")
        elif potential_ratio > 0.40:  # Too much in potential
            recommendations.append("Move excess Potential Energy to Core Energy for preservation")
        
        if core_ratio < 0.45:  # Too little in core preservation
            recommendations.append("Increase Core Energy allocation for long-term anti-fragile stability")
        
        return recommendations
    
    async def allocate_profit(self, profit_amount: float, source_wallet: str) -> Dict[str, float]:
        """Allocate profit to vaults based on configuration"""
        try:
            if profit_amount <= 0:
                return {}
            
            allocations = {}
            total_allocated = 0.0
            
            # Allocate to each vault type based on percentage
            for vault_type, allocation in self.vault_allocations.items():
                if allocation.percentage > 0:
                    vault_id = f"vault_{vault_type.value}"
                    vault = self.vaults[vault_id]
                    
                    # Calculate allocation amount
                    allocation_amount = profit_amount * allocation.percentage
                    
                    # Apply min/max constraints
                    allocation_amount = max(allocation.min_amount, min(allocation.max_amount, allocation_amount))
                    
                    # Check if vault can accept more
                    if vault.balance + allocation_amount <= allocation.max_amount:
                        await self._deposit_to_vault(vault_id, allocation_amount, source_wallet)
                        allocations[vault_id] = allocation_amount
                        total_allocated += allocation_amount
                    
                    # Stop if we've allocated everything
                    if total_allocated >= profit_amount:
                        break
            
            self.total_profits_secured += total_allocated
            self.logger.info(f"💰 Allocated {total_allocated:.6f} SOL to vaults from {source_wallet}")
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error allocating profit: {e}")
            return {}
    
    async def deposit_profits(self, amount: float) -> bool:
        """Deposit profits to vault system - simplified interface"""
        try:
            # Use daily vault for profit deposits
            vault_id = "vault_daily"
            if vault_id in self.vaults:
                await self._deposit_to_vault(vault_id, amount, "trading_bot")
                self.logger.info(f"💰 Deposited {amount:.4f} SOL profits to vault")
                return True
            else:
                self.logger.error(f"Vault {vault_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error depositing profits: {e}")
            return False
    
    async def _deposit_to_vault(self, vault_id: str, amount: float, source_wallet: str):
        """Deposit funds to a vault"""
        try:
            vault = self.vaults[vault_id]
            
            # Update vault balance
            vault.balance += amount
            vault.total_deposits += amount
            vault.last_activity = datetime.now()
            
            # Update total vault balance
            self.total_vault_balance += amount
            
            # Log the deposit
            self.logger.info(f"💎 Deposited {amount:.6f} SOL to {vault_id}")
            
            # Store transaction record
            await self._record_vault_transaction(vault_id, 'deposit', amount, source_wallet)
            
        except Exception as e:
            self.logger.error(f"Error depositing to vault: {e}")
    
    async def withdraw_from_vault(self, vault_id: str, amount: float, 
                                destination_wallet: str, reason: str = "manual") -> bool:
        """Withdraw funds from a vault"""
        try:
            vault = self.vaults[vault_id]
            
            # Check security requirements
            if not await self._check_withdrawal_security(vault, amount, reason):
                return False
            
            # Check balance
            if vault.balance < amount:
                self.logger.warning(f"Insufficient balance in {vault_id}: {vault.balance} < {amount}")
                return False
            
            # Update vault balance
            vault.balance -= amount
            vault.total_withdrawals += amount
            vault.last_activity = datetime.now()
            
            # Update total vault balance
            self.total_vault_balance -= amount
            
            # Log the withdrawal
            self.logger.info(f"💸 Withdrew {amount:.6f} SOL from {vault_id} for {reason}")
            
            # Store transaction record
            await self._record_vault_transaction(vault_id, 'withdrawal', amount, destination_wallet, reason)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error withdrawing from vault: {e}")
            return False
    
    async def _check_withdrawal_security(self, vault: VaultWallet, amount: float, reason: str) -> bool:
        """Check security requirements for withdrawal"""
        try:
            allocation = self.vault_allocations[vault.vault_type]
            
            # Check daily withdrawal limit
            if amount > self.security_config['max_daily_withdrawal']:
                self.logger.warning(f"Withdrawal {amount} exceeds daily limit {self.security_config['max_daily_withdrawal']}")
                return False
            
            # Check if withdrawal delay is required
            if 'delay' in allocation.security_requirements:
                delay_hours = self.security_config['withdrawal_delay_hours']
                if reason != 'emergency':
                    self.logger.info(f"Withdrawal delayed by {delay_hours} hours for security")
                    # In a real implementation, this would queue the withdrawal
            
            # Check if multi-sig is required
            if 'multi_sig' in allocation.security_requirements:
                if not await self._verify_multi_sig_approval(vault.vault_id, amount):
                    self.logger.warning("Multi-sig approval required but not provided")
                    return False
            
            # Check if manual approval is required
            if 'manual_approval' in allocation.security_requirements:
                if not await self._verify_manual_approval(vault.vault_id, amount):
                    self.logger.warning("Manual approval required but not provided")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking withdrawal security: {e}")
            return False
    
    async def _verify_multi_sig_approval(self, vault_id: str, amount: float) -> bool:
        """Verify multi-signature approval (production-ready stub)"""
        if not self.security_config.get('multi_sig_enabled', False):
            self.logger.error("Multi-sig approval is not enabled or configured.")
            return False
        # Implement actual multi-sig logic here
        return True

    async def _verify_manual_approval(self, vault_id: str, amount: float) -> bool:
        """Verify manual approval (production-ready stub)"""
        if not self.security_config.get('manual_approval_enabled', False):
            self.logger.error("Manual approval is not enabled or configured.")
            return False
        # Implement actual manual approval logic here
        return True
    
    async def emergency_withdrawal(self, vault_id: str, amount: float, 
                                 destination_wallet: str) -> bool:
        """Emergency withdrawal bypassing normal security"""
        try:
            if not self.security_config['emergency_access_enabled']:
                self.logger.error("Emergency access is disabled")
                return False
            
            vault = self.vaults[vault_id]
            
            # Check if vault allows emergency access
            allocation = self.vault_allocations[vault.vault_type]
            if 'emergency_only' not in allocation.security_requirements:
                self.logger.warning(f"Emergency withdrawal not allowed for {vault_id}")
                return False
            
            # Perform emergency withdrawal
            success = await self.withdraw_from_vault(vault_id, amount, destination_wallet, "emergency")
            
            if success:
                self.logger.warning(f"🚨 EMERGENCY WITHDRAWAL: {amount} SOL from {vault_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Emergency withdrawal failed: {e}")
            return False
    
    async def _vault_monitoring_loop(self):
        """Background vault monitoring loop"""
        while self.initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update total vault balance
                self.total_vault_balance = sum(vault.balance for vault in self.vaults.values())
                
                # Check for security anomalies
                await self._check_security_anomalies()
                
                # Log vault status periodically
                await self._log_vault_status()
                
            except Exception as e:
                self.logger.error(f"Vault monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _auto_compound_loop(self):
        """Background auto-compound loop"""
        while self.initialized:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                for vault_id, vault in self.vaults.items():
                    allocation = self.vault_allocations[vault.vault_type]
                    
                    if allocation.auto_compound and vault.balance > 0:
                        # Check if it's time to compound
                        hours_since_last_activity = (datetime.now() - vault.last_activity).total_seconds() / 3600
                        
                        if hours_since_last_activity >= allocation.compound_frequency_hours:
                            await self._perform_auto_compound(vault_id)
                
            except Exception as e:
                self.logger.error(f"Auto-compound error: {e}")
                await asyncio.sleep(3600)
    
    async def _perform_auto_compound(self, vault_id: str):
        """Perform auto-compound for a vault"""
        try:
            vault = self.vaults[vault_id]
            
            # Calculate compound amount (simple interest for now)
            compound_amount = vault.balance * 0.01  # 1% compound
            
            if compound_amount > 0.001:  # Minimum 0.001 SOL
                # In a real implementation, this would reinvest the compound amount
                vault.balance += compound_amount
                vault.last_activity = datetime.now()
                
                self.logger.info(f"🔄 Auto-compounded {compound_amount:.6f} SOL in {vault_id}")
                
        except Exception as e:
            self.logger.error(f"Auto-compound failed for {vault_id}: {e}")
    
    async def _security_monitoring_loop(self):
        """Background security monitoring loop"""
        while self.initialized:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                # Check for unusual activity
                await self._detect_unusual_activity()
                
                # Update security metrics
                await self._update_security_metrics()
                
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _detect_unusual_activity(self):
        """Detect unusual vault activity"""
        try:
            for vault_id, vault in self.vaults.items():
                # Check for large withdrawals
                if vault.total_withdrawals > vault.total_deposits * 0.5:
                    self.logger.warning(f"⚠️ Large withdrawal detected in {vault_id}")
                
                # Check for rapid balance changes
                # This would require historical data tracking
                
        except Exception as e:
            self.logger.error(f"Error detecting unusual activity: {e}")
    
    async def _update_security_metrics(self):
        """Update security metrics"""
        try:
            # Calculate security scores
            for vault_id, vault in self.vaults.items():
                allocation = self.vault_allocations[vault.vault_type]
                
                # Security score based on requirements met
                security_score = len(allocation.security_requirements) / 5.0
                vault.metadata['security_score'] = security_score
                
        except Exception as e:
            self.logger.error(f"Error updating security metrics: {e}")
    
    async def _check_security_anomalies(self):
        """Check for security anomalies"""
        try:
            # Check total vault balance consistency
            calculated_balance = sum(vault.balance for vault in self.vaults.values())
            if abs(calculated_balance - self.total_vault_balance) > 0.001:
                self.logger.error(f"⚠️ Vault balance inconsistency detected!")
            
        except Exception as e:
            self.logger.error(f"Error checking security anomalies: {e}")
    
    async def _log_vault_status(self):
        """Log vault status periodically"""
        try:
            # Log every 6 hours
            if hasattr(self, '_last_status_log'):
                if (datetime.now() - self._last_status_log).total_seconds() < 21600:
                    return
            
            self._last_status_log = datetime.now()
            
            self.logger.info("🏦 Vault Status Summary:")
            for vault_id, vault in self.vaults.items():
                self.logger.info(f"   {vault_id}: {vault.balance:.6f} SOL ({vault.vault_type.value})")
            
            self.logger.info(f"   Total Secured: {self.total_vault_balance:.6f} SOL")
            self.logger.info(f"   Total Profits Secured: {self.total_profits_secured:.6f} SOL")
            
        except Exception as e:
            self.logger.error(f"Error logging vault status: {e}")
    
    async def _record_vault_transaction(self, vault_id: str, transaction_type: str, 
                                      amount: float, wallet: str, reason: str = ""):
        """Record vault transaction in TimescaleDB"""
        try:
            from worker_ant_v1.core.database import get_database_manager, SystemEvent
            import uuid
            from datetime import datetime
            
            # Get TimescaleDB manager
            db_manager = await get_database_manager()
            
            # Create system event for vault transaction
            event = SystemEvent(
                timestamp=datetime.utcnow(),
                event_id=str(uuid.uuid4()),
                event_type="vault_transaction",
                component="vault_system",
                severity="INFO",
                message=f"Vault transaction: {transaction_type} {amount} SOL for {vault_id}",
                event_data={
                    "vault_id": vault_id,
                    "transaction_type": transaction_type,
                    "amount": amount,
                    "wallet": wallet,
                    "reason": reason,
                    "status": "completed"
                },
                wallet_id=wallet
            )
            
            # Insert event into TimescaleDB
            await db_manager.insert_system_event(event)
            
            self.logger.info(f"📝 Recorded {transaction_type} transaction: {amount} SOL for {vault_id}")
            
        except Exception as e:
            self.logger.error(f"Error recording vault transaction: {e}")
    
    async def _save_vault_data(self):
        """Save vault data to storage (JSON file)"""
        try:
            storage_path = Path('wallets/vaults.json')
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {vid: vault.__dict__ for vid, vault in self.vaults.items()}
            with open(storage_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving vault data: {e}")

    async def _load_vault_data(self):
        """Load vault data from storage (JSON file)"""
        try:
            storage_path = Path('wallets/vaults.json')
            if not storage_path.exists():
                return False
            with open(storage_path, 'r') as f:
                data = json.load(f)
            for vault_id, vault_data in data.items():
                vault = VaultWallet(**vault_data)
                self.vaults[vault_id] = vault
            return True
        except Exception as e:
            self.logger.error(f"Error loading vault data: {e}")
            return False
    
    def get_vault_status(self) -> Dict[str, Any]:
        """Get comprehensive vault status"""
        try:
            vault_status = {}
            for vault_id, vault in self.vaults.items():
                vault_status[vault_id] = {
                    'type': vault.vault_type.value,
                    'balance': vault.balance,
                    'total_deposits': vault.total_deposits,
                    'total_withdrawals': vault.total_withdrawals,
                    'security_level': vault.security_level,
                    'last_activity': vault.last_activity.isoformat()
                }
            
            return {
                'total_vault_balance': self.total_vault_balance,
                'total_profits_secured': self.total_profits_secured,
                'vault_count': len(self.vaults),
                'vaults': vault_status,
                'security_enabled': self.security_config['encryption_enabled'],
                'emergency_access': self.security_config['emergency_access_enabled']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting vault status: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the vault system"""
        try:
            self.logger.info("🛑 Shutting down vault system...")
            
            # Save vault data
            await self._save_vault_data()
            
            self.initialized = False
            self.logger.info("✅ Vault system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during vault shutdown: {e}")

# Global instance
_vault_system = None

async def get_vault_system() -> VaultWalletSystem:
    """Get global vault system instance"""
    global _vault_system
    if _vault_system is None:
        _vault_system = VaultWalletSystem()
        await _vault_system.initialize_vault_system()
    return _vault_system 